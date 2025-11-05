from enum import Enum
import os
import json

from datetime import date, datetime
from pydantic import BaseModel, Field, validate_call
from typing import Annotated, List, TypedDict
from data_schemas import CreateEvent, CreateTask

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ChatMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command, interrupt

from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from data_schemas import CreateEvent, CreateTask, Email
import calendar_tools
import utils

os.environ["GOOGLE_API_KEY"] = utils.get_json_field('config.json', 'gemini_key')

SYSTEM_PROMPT = f"""You are a helpful AI agent responsible for managing a user's Google Calendar.
An AI chatbot is talking to the user and will relay requests to you in order to help the user. Your goal is to fulfill chatbot's requests using ONLY the tools provided to you. 
You should think step by step and decide when to use tools to take action. Ensure you reason in every response.
Respond in this format:

Thought: [Your reasoning]
Action: [The structured tool call JSON objects]

INFO:
Today's date is {calendar_tools.today}

RULES:
- If a request is vague ask for more information.
- You MUST check user availability before scheduling events and tasks. If there are conflicts, inform the chatbot with the exact times of conflicts and hold off on scheduling tasks. 
- Give dates in RFC 3339 format.
- You MUST use the `Respond` tool to respond to the chatbot with a request completion notification or the information they requested when you are done."""

REQUEST_STATUS = Enum("REQUEST_STATUS", { "FAILURE": "failure", "SUCCESS": "success" })

@tool
@validate_call
def search_user_availability(
        start: datetime = Field(..., description="Start of date of the search range in ISO 8601 format."), 
        end: datetime = Field(..., description="End of date of the search range in ISO 8601 format.")
    ):
    """Use this to check the user's availability.
Instructions: Enter the start and end times to search. You will receive a list of event objects back.\n Event object important fields:\n 1. summary - name of the event.\n 2. start - start time and timezone of the event.\n 3. end - end time and timezone of the event."""
    return json.dumps({
        "status": "success",
        "summary": "fetched events and tasks in specified range",
        "events": calendar_tools.get_events_in_range(start,end), 
        "tasks": calendar_tools.get_tasks_in_range(start,end)
    }, indent=2)

@tool
@validate_call
def change_event_time(
        event_id: str = Field(..., description="ID of the event provided by the Google."),
        start: datetime = Field(..., description="The new start time of the event in ISO 8601 format."),
        end: datetime = Field(..., description="The new end time of the event in ISO 8601 format.")
    ):
    """Use this to reschedule an event.
Instructions: Enter the Google Services API provided ID. If you don't have it, first search for the event using search_user_availability and obtain the ID from the tool."""
    return calendar_tools.reschedule_event(event_id, start, end)

@tool
@validate_call
def cancel_event(
        event_id: str = Field(..., description="ID of the event provided by Google.")
    ):
    """Use this tool to delete an event.
Instructions:  Enter the Google Service API provided ID. If you don't have it, first search for the event using search_user_availability and obtain the ID from the tool."""
    return calendar_tools.remove_event(event_id)

@tool
@validate_call
def change_task_time(
        task_id: str = Field(..., description="ID of the task provided by Google."),
        due: datetime = Field(..., description="The new due time of the event in ISO 8601 format.")
    ):
    """Use this tool to change the time of a task.
Instructions: Enter the Google Services API provided ID of the task you want to reschedule along with the new due time. If you don't have it, first search for the task using search_user_availability and obtain the ID from the tool."""
    return calendar_tools.reschedule_task(task_id, due)

@tool
@validate_call
def cancel_task(
        task_id: str = Field(..., description="ID of the task provided by Google.")
    ):
    """Use this tool to delete a task.
Instructions: Enter Google Services API provided ID for the task you want to cancel. If you don't have it, first search for the task using search_user_availability and obtain the ID from the tool."""
    return calendar_tools.remove_task(task_id)


def create_task(state):
    call = state['messages'][-1].tool_calls[0]
    task = CreateTask.model_validate(call['args'])
    return {"messages": state['messages'] + [ToolMessage(tool_call_id=call['id'], content=calendar_tools.add_task(task))] }

def create_event(state):
    call = state['messages'][-1].tool_calls[0]
    event = CreateEvent.model_validate(call['args'])
    return {"messages": state['messages'] + [ToolMessage(tool_call_id=call['id'], content=calendar_tools.add_event(event))]}

class AskQuestion(BaseModel):
    """Used to ask the chatbot for more information or clarification about the request."""
    question: str

class Respond(BaseModel):
    """Used to respond back to the chatbot with requested information and indicate task completion."""
    
    request_status: REQUEST_STATUS = Field(..., description="Whether the task was successfully fulfilled or ended in failure.")
    summary: str = Field(..., description="A SHORT summary of what you did during the task.")
    information: str = Field(None, description="The information the chatbot requested in detail. Only use if the chatbot requested information and the task ended in success.")
    error_info: str = Field(None, description="A summary of the issues that arose during the task if it ended in failure.")



tools = [change_event_time, cancel_event, change_task_time, cancel_task, search_user_availability]
tool_node = ToolNode(tools)

summarizer = ChatGoogleGenerativeAI(model='gemini-2.5-pro')

c_agent = summarizer.bind_tools(tools + [AskQuestion, Respond, CreateTask, CreateEvent])


def ask_question(state):
    call = state['messages'][-1].tool_calls[0]
    tool_call_id = call["id"]
    q = AskQuestion.model_validate(call["args"])
    
    
    tool_message = ToolMessage(tool_call_id=tool_call_id, content=interrupt(q.question))
  
    return {"messages": state["messages"] + [tool_message]}

def respond(state):
    call = state['messages'][-1].tool_calls[0]
    tool_call_id = call["id"]
    r = Respond.model_validate(call["args"])

    tool_message = ToolMessage(tool_call_id=tool_call_id, content=json.dumps(r.model_dump(mode='json')))

    return {"messages": state["messages"] + [tool_message]}



def route(state):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        state["messages"] = state["messages"] + [SystemMessage(content="You MUST call a tool")]
        return "agent"
    
    if last_message.tool_calls[0]['name'] == "AskQuestion":
        return "ask_question"
    elif last_message.tool_calls[0]['name'] == "Respond":
        return "respond"
    elif last_message.tool_calls[0]['name'] == "CreateTask":
        return "create_task"
    elif last_message.tool_calls[0]['name'] == "CreateEvent":
        return "create_event"
    else:
        return "action"
    
def call_agent(state):
    messages = state["messages"]

    response = c_agent.invoke(messages)
    
    
    return {"messages": messages + [response]}

workflow = StateGraph(MessagesState)


workflow.add_node("agent", call_agent)
workflow.add_node("action", tool_node)
workflow.add_node("respond", respond)
workflow.add_node("ask_question", ask_question)
workflow.add_node("create_task", create_task)
workflow.add_node("create_event", create_event)
workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    route,
    path_map=["action", "respond", "ask_question", "create_task", "create_event", "agent"]
)

workflow.add_edge("action", "agent")
workflow.add_edge("respond", END)
workflow.add_edge("ask_question", "agent")
workflow.add_edge("create_task", "agent")
workflow.add_edge("create_event", "agent")

memory = InMemorySaver()

app = workflow.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "1"}}

def start_workflow(request: str = None, answer: str = None):

    if answer:
        events = list(app.stream(
            Command(resume=answer),
            config=config,
            stream_mode="values"
        ))
    else:
        events = list(app.stream(
            {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=request)
                ]
            },
            config,
            stream_mode="values"
        ))


    if len(app.get_state(config).next) > 0 and app.get_state(config).next[0] == 'ask_question':
        return (events[-1]['messages'][-1].tool_calls[0]['args']['question'],True)
    else:
        memory.storage.clear()
        
        return (events[-1]['messages'][-1].content, False)
    
# inp = None

#while True:
    #res = start_workflow("""{
    #"task": "Give me the user's availability for this entire week and cancel any events with their mom."
#}""", answer=inp)
   # if res[1]:
   #     inp = input(res[0])
  #      continue
  #  else:
   #     break
    
