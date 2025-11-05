import base64
import re
import threading
import time
import json

from datetime import date, datetime
from urllib.parse import urlparse
from dateutil import parser
from pydantic import BaseModel

import pytz
import iso8601
import utils

from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from data_schemas import CreateTask, CreateEvent

g_cal = None
g_tasks = None
default_tasklist = None
time_zone = None
today = None

def init_tasks(creds):
    global g_tasks, default_tasklist
    g_tasks = build('tasks', 'v1', credentials=creds)
    tasklists = g_tasks.tasklists().list().execute().get('items', [])
    if tasklists: default_tasklist = tasklists[0]['id']

def init_calendar(creds):
    global g_cal, time_zone, today
    g_cal = build('calendar', 'v3', credentials=creds)
    tz = g_cal.calendarList().get(calendarId='primary').execute()
    time_zone = pytz.timezone(tz['timeZone'])
    today = datetime.now(time_zone)


def add_task(t: CreateTask) -> str:
    global g_tasks

    try:
        task_result = g_tasks.tasks().insert(
            tasklist=default_tasklist,
            body=t.model_dump(mode='json', exclude_none=True)
        ).execute()
    except HttpError as error:
        return json.dumps({"status": "failure", "result": "failed to create task", "error": error})

    return json.dumps({
        "status": "success",
        "result": "created task",
        "task_id": task_result['id'],
        "due": t.due.isoformat(),
    }, indent=2)

def reschedule_task(task_id: str, due: datetime) -> str:
    global g_tasks

    try:
        task_result = g_tasks.tasks().patch(
            tasklist=default_tasklist,
            task=task_id,
            body={
                "due": time_zone.localize(due).isoformat() if is_naive(due) else due.isoformat()
            }
        ).execute()
    except HttpError as error:
        return json.dumps({"status": "failure", "result": "failed to reschedule task", "error": error})

    return json.dumps({
        "status": "success",
        "result": "updated task time",
        "task_id": task_id,
        "new_due": due.isoformat()
    }, indent=2)

def remove_task(task_id: str) -> str:
    global g_tasks
    
    try:
        g_tasks.tasks().delete(
            taskslist=default_tasklist,
            task=task_id
        ).execute()
    except HttpError as error:
        return json.dumps({"status": "failure", "result": "failed to remove task", "error": error})

    return json.dumps({"status": "success","result": "removed task", "id_of_task_removed": task_id}, indent=2)


def add_event(e: CreateEvent) -> str:
    global g_cal

    try:
        event_result = g_cal.events().insert(
            calendarId="primary",
            body=e.model_dump(mode='json', exclude_none=True)
        ).execute()
    except HttpError as error:
        return json.dumps({
            "status": "failure",
            "result": "failed to add event",
            "error": error
        }, indent=2)

    return json.dumps({
        "status": "success",
        "result": "created event",
        "event_id": event_result['id'],
        "start": e.start.dateTime.isoformat(),
        "end": e.end.dateTime.isoformat(),
    }, indent=2)

def reschedule_event(event_id: str, start: datetime, end: datetime) -> str:
    global g_cal

    try:
        g_cal.events().patch(
            calendarId="primary",
            eventId=event_id,
            body={
                "start": {
                    "dateTime": time_zone.localize(start).isoformat() if is_naive(start) else start.isoformat()
                },
                "end": {
                    "dateTime": time_zone.localize(end).isoformat() if is_naive(end) else end.isoformat()
                }
            }
        ).execute()
    except HttpError as error:
        return json.dumps({"status": "failure", "result": "failed to reschedule event", "error": error})

    return json.dumps({
        "status": "success",
        "result": "updated event time",
        "event_id": event_id,
        "new_start": start.isoformat(),
        "new_end": end.isoformat()
    }, indent=2)


def remove_event(event_id: str):
    global g_cal
    
    try:
        g_cal.events().delete(
            calendarId='primary',
            eventId=event_id
        ).execute()
    except HttpError as error:
        return json.dumps({"status": "failure", "result": "failed to remove event", "error": error})

    return json.dumps({"status": "success","result": "removed event", "id_of_event_removed": event_id}, indent=2)



def is_naive(dt: datetime):
    return dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None

def get_events_in_range(start: datetime, end: datetime) -> list:
    global g_cal

    try:
        events_result = g_cal.events().list(
            calendarId='primary', timeMin=time_zone.localize(start).isoformat() if is_naive(start) else start.isoformat(),
            timeMax=time_zone.localize(end).isoformat() if is_naive(end) else end.isoformat(),
            maxResults=20, singleEvents=True,
            orderBy='startTime'
        ).execute()
    except HttpError as error:
        print(error)
        return [error]

    events = events_result.get('items', [])

    event_list = []
    for e in events:
        data = {
            "id": e['id'],
            "start": e['start'],
            "end": e['end'],
            "summary": e['summary'],
            "description": e['description'][:30] if 'description' in e else None,
            "location": e.get('location', None),
            "recurrence": e.get('recurrence', None),
            "reminders": e.get('reminders', None),
            "attendees": e.get('attendees', None)
        }

        if data['attendees'] and len(data['attendees']) > 5:
            del data['attendees']

        cleaned_data = {k: v for k, v in data.items() if v is not None}
        event_list.append(cleaned_data)

    return event_list


def get_tasks_in_range(start: datetime, end: datetime) -> list:
    global g_tasks

    try:
        results = g_tasks.tasklists().list(maxResults=10).execute()
    except HttpError as error:
        return [error]
    tasklists = results.get('items', [])

    tasks_in_range = []
    for tasklist in tasklists:
        try:
            tasks_results = g_tasks.tasks().list(
                tasklist=tasklist['id'],
                showCompleted=False,
                showDeleted=False,
                maxResults=10
            ).execute()
        except HttpError as error:
            return [error]

        items = tasks_results.get('items', [])
        
        for task in items:
            due_time = datetime.fromisoformat(task['due']).replace(tzinfo=None)
            
            due_time = time_zone.localize(due_time)
            
            if start <= due_time <= end:
                tasks_in_range.append(task)

    return tasks_in_range


init_calendar(utils.creds)
init_tasks(utils.creds)

