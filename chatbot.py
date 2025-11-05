import os
import json

from datetime import date, datetime
from pydantic import BaseModel, Field, validate_call
from typing import Optional, TypedDict
import re


import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import ChatMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from data_schemas import CreateEvent, CreateTask, Email
import calendar_tools
import calendar_agent
import gmail_agent
import utils
from enum import Enum
end = False

class AgentType(str, Enum):
    CALENDAR_AGENT = "calendar_agent"
    GMAIL_AGENT = "gmail_agent"
   
class ConfirmRequestCompletion(BaseModel):
    """Use this to send the user a message indicating their request was completed. 
Instructions: After completing all of the user's requests, send a confirmation message to them using this tool. DO NOT ask follow-up questions."""
    message: str

class GiveUserInfo(BaseModel):
    """Use this to send the user useful information. 
Instructions: Enter a string containing the information you want to give to the user. This string should not contain questions. If you want to give the user information and then a follow-up question, call GiveUserInfo and then PromptUser in succession."""
    info: str

class PromptUser(BaseModel):
    """Use this to receive user input. 
Instructions: Enter a question to receive input from the user."""
    prompt: str


def give_user_info(args: dict):
    info = GiveUserInfo.model_validate(args)
    print("Assistant\n" + info.info, end='\n\n')


    return "User received provided information."

def confirm_request_completion(args: dict):
    message = ConfirmRequestCompletion.model_validate(args)
    print("Assistant\n" + message.message, end='\n\n')

    return "User received request completion confirmation."

def prompt_user(args: dict):
    ask = PromptUser.model_validate(args)

    response = input("Assistant\n" + ask.prompt + '\n\nUser\n')
    print('')
    resp = f'The user replied \"{response}\"'

    return resp

tool_map = {
    "ConfirmRequestCompletion": confirm_request_completion,
    "GiveUserInfo": give_user_info,
    "PromptUser": prompt_user
}


def make_email_prompt(inject: str):
    return f"""You are a helpful AI chatbot responsible for accessing a user's Google Calendar and Gmail.
Your goal is to receive requests from the user and give instructions to the Google Calendar Agent and Gmail Agent to complete them.
The Gmail Agent handles searching for messages, labeling emails, and creating new emails.
The Calendar Agent handles scheduling, deleting, and rescheduling events, as well as looking up availability.
You should think step by step and decide when to use tools to take action. Ensure you reason in every response.
Respond in this format:

Thought: [Your reasoning]
Action: [The structured tool call JSON objects]

Task:
The user has received an email. Here is a structured breakdown of the contents.

{inject}

1. Let the user know about the information contained in the email in 1 to 2 sentences.
2. Ask the user what action they want you to take.
3. Complete any tasks that the user requests from you.

Info:
- The date is {calendar_tools.today.isoformat()}.
- The timezone is {calendar_tools.time_zone.zone}.

Rules:
- The calendar agent cannot access the user's emails. DO NOT tell the user you can do this for them.
- Be specific in your requests to the Calendar Agent. Give dates, times, and names whenever possible.
- The Calendar Agent saves no memory of conversation between calls, so make sure you give context wherever possible.
- Give datetimes to the Calendar Agent in RFC 3339 format and give dates to the user in [month name] [day] format with the time specified in AM or PM.
- NEVER talk to the user in plain text.You may ONLY use the `PromptUser`, `GiveUserInfo`, and `ConfirmRequestCompletion` tools to speak to the user. Use CallCalendarAgent
- Ensure the user doesn't have any more requests before calling `ConfirmRequestCompletion`.

- When a user asks to create a new email label, ENSURE you ask them what the description should be, and give that information to the Gmail Agent.
- When a user asks you to add an email to a label category, ENSURE you provide the Gmail Agent with the email's ID and the name of the label they should add it to.
"""


class CallAgent(BaseModel):
    """Use this to send either the Calendar Agent or Gmail Agent a task to complete for you. 
Gmail Agent: handles email searches and email tagging.
Calendar Agent: handles event and task lookup as well as scheduling, rescheduling, and removing tasks and events"""
    agent_type: AgentType = Field(..., description="The type of agent you are picking")
    task: str = Field(..., description="The detailed description task to complete.")
    context: Optional[str] = Field(None, description="Extra context that the Agent may need to complete the task. (i.e. timezones, names, emails)")


class RespondToAgent(BaseModel):
    """Used to respond to questions from the Calendar Agent."""
    answer: str = Field(..., description="The information that answers the Calendar Agent's question.")


def talk_to_user(state):
    global end
    last_message = state["messages"][-1]
    
    responses = []
    for i in last_message.tool_calls:
        resp = tool_map[i['name']](i['args'])
        if i['name'] == "ConfirmRequestCompletion":
            end = True
        responses.append(ToolMessage(tool_call_id=i['id'], content=resp))

        if re.search(r'label', resp, re.IGNORECASE):
            responses.append(SystemMessage(content="If users ask to create a label ensure you ask for their label description.\nIf they ask you to add an email to a label, provide the gmail agent with the email id and label name."))

    return {"messages": state["messages"] + responses}


def route_agent(state):
    messages = state["messages"]
    last_message = messages[-1]

    
    if last_message.tool_calls[0]['name'] in tool_map:
        return "talk_to_user"
    elif last_message.tool_calls[0]['name'] == "CallAgent":
        if last_message.tool_calls[0]['args']['agent_type'] == "calendar_agent":
            return "call_calendar_agent"
        else:
            return "call_gmail_agent"
        

    print("something wrong happened", last_message.tool_calls)

    return END


def should_continue(state):
    if end:
        return END
    else:
        return "chatbot"
    



summarizer = ChatGoogleGenerativeAI(model='gemini-2.5-pro')
chatbot = summarizer.bind_tools([PromptUser, GiveUserInfo, ConfirmRequestCompletion, CallAgent])
answerer = summarizer.with_structured_output(RespondToAgent)

def call_calendar_agent(state):
    call = state["messages"][-1].tool_calls[0]
    agent_args = CallAgent.model_validate(call['args'])
    res = calendar_agent.start_workflow(request=agent_args.model_dump_json(indent=2, exclude={'agent_type'}))
    while res[1]:
        answer = answerer.invoke(state['messages'] + [HumanMessage(content=res[0])])
        res = calendar_agent.start_workflow(answer=answer)
    
    return {"messages": state['messages'] + [ToolMessage(tool_call_id=call['id'], content=res[0])]}


def call_gmail_agent(state):
    call = state["messages"][-1].tool_calls[0]
    print(call)
    agent_args = CallAgent.model_validate(call['args'])
    res = gmail_agent.start_workflow(request=agent_args.model_dump_json(indent=2, exclude={'agent_type'}))
    while res[1]:
        answer = answerer.invoke(state['messages'] + [HumanMessage(content=res[0])])
        res = calendar_agent.start_workflow(answer=answer)
    
    return {"messages": state['messages'] + [ToolMessage(tool_call_id=call['id'], content=res[0])]}
    
def call_chatbot(state):
    messages = state['messages']


    resp = chatbot.invoke(messages)


    return {"messages": state['messages'] + [resp]}


workflow = StateGraph(MessagesState)

workflow.add_node("chatbot", call_chatbot)
workflow.add_node("talk_to_user", talk_to_user)
workflow.add_node("call_calendar_agent", call_calendar_agent)
workflow.add_node("call_gmail_agent", call_gmail_agent)

workflow.add_edge(START, "chatbot")

workflow.add_conditional_edges(
    # First, we define the start node. We use `chatbot`.
    # This means these are the edges taken after the `chatbot` node is called.
    "chatbot",
    # Next, we pass in the function that will determine which node is called next.
    route_agent,
    path_map=["talk_to_user", "call_calendar_agent", "call_gmail_agent"]
)

workflow.add_conditional_edges(
    "talk_to_user",
    should_continue,
    path_map=["chatbot", END]
)

workflow.add_edge("call_calendar_agent", "chatbot")
workflow.add_edge("call_gmail_agent", "chatbot")

app = workflow.compile()


config = {"configurable": {"thread_id": "1"}}


def start_new_email_agent(email):
    return app.stream(
        {
            "messages": [
                SystemMessage(content=make_email_prompt(email)),
                {
                    "role" : "human", 
                    "content": "Help the user with the new message."
                }
            ]
        },
        config,
        stream_mode="values"
    )

def make_conversation_prompt():
    return f"""You are a helpful AI chatbot responsible for accessing a user's Google Calendar and Gmail.
Your goal is to receive requests from the user and give instructions to the Google Calendar Agent and Gmail Agent to complete them.
You should think step by step and decide when to use tools to take action. Ensure you reason in every response.
Respond in this format:

Thought: [Your reasoning]
Action: [The structured tool call JSON objects]

Task:
The user has opened a chat conversation with you.

1. Ask the user what they would like you to do.
2. Complete their request if it can be performed using any of your tools. If not, inform the user and STOP.
3. Ask the user if they need anything else, and if not, STOP.

Info:
- The date is {calendar_tools.today.isoformat()}.
- The timezone is {calendar_tools.time_zone.zone}.

Rules for Calendar Agent:
- Be specific in your requests to the Calendar Agent. Give dates, times, and names whenever possible.
- The Calendar Agent saves no memory of conversation between calls, so make sure you give context wherever possible.
- Give datetimes to the Calendar Agent in RFC 3339 format and give dates to the user in [month name] [day] format with the time specified in AM or PM.
- NEVER talk to the user in plain text.You may ONLY use the `PromptUser`, `GiveUserInfo`, and `ConfirmRequestCompletion` tools to speak to the user. Use CallCalendarAgent
- Ensure the user doesn't have any more requests before calling `ConfirmRequestCompletion`.

Rules for Gmail Agent:
- The gmail agent cannot send emails.
- Use dates and times whenever posisble
"""

def start_new_conversation_agent():
    return app.stream(
        {
            "messages": [
                SystemMessage(content=make_conversation_prompt()),
                {
                    "role" : "human", 
                    "content": "Help the user with the new message."
                }
            ]
        },
        config,
        stream_mode="values"
    )


