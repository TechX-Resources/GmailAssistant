import base64
import json
import re
import threading
import time
import unicodedata
from data_schemas import Email
import utils
import semantics
from typing import Callable, Union
from datetime import date
from urllib.parse import urlparse
from dateutil import parser

from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError

gmail = None
label_descriptors = utils.get_json_field('config.json', 'user_labels')
map_of_labels = { label['name']: label['id'] for label in label_descriptors }
ids_to_names = { v: k for k, v in map_of_labels.items()}
user_email = None


def load_map_of_labels():
    global map_of_labels
    map_of_labels = { label['name']: label['id'] for label in label_descriptors }

def add_label_descriptor(label_name, label_id, description):
    global label_descriptors
    label_descriptors.append({
        "name": label_name,
        "id": label_id,
        description: description
    })



def check_label_ids():
    response = gmail.users().labels().list(userId='me').execute()
    labels = response.get('labels', [])
    label_set = set([label['name'] for label in labels])
    for k in map_of_labels:
        if k not in label_set:
            id = create_label(k)
            map_of_labels[k] = id
            for i in range(len(label_descriptors)):
                if label_descriptors[i]['name'] == k:
                    label_descriptors[i]['id'] = id

    utils.update_json('config.json', 'user_labels', label_descriptors)


def add_email_label(email_id: str, label_name: str):
    
    label_id = map_of_labels[label_name]

    try:
        gmail.users().messages().modify(
            userId='me',
            id=email_id,
            body={
                'addLabelIds': [label_id]
            }
        ).execute()
    except HttpError as error:
        return json.dumps({
            "status": "failure",
            "error": str(error)
        })
    
    return json.dumps({
        "status": "success",
        "result": "added label to email",
        "label_name": label_name,
        "email_id": email_id
    })
    


def remove_email_label(email_id: str, label_name: str):
    label_id = map_of_labels[label_name]
    try:
        gmail.users().messages().modify(
            userId='me',
            id=email_id,
            body={
                'removeLabelIds': [label_id]
            }
        ).execute()
    except HttpError as error:
        return json.dumps({
            "status": "failure",
            "error": error
        })
    
    try: 
        return json.dumps({
            "status": "success",
            "result": "removed label from email",
            "label_name": str(label_name),
            "email_id": email_id
        })
    except Exception as e:
        print(e)

def create_label(label_name: str, description:str=None , color=None):
    label_body = {
        "name": label_name,
        "labelListVisibility": "labelShow",     
        "messageListVisibility": "show",        
    }

    if color:
        label_body["color"] = {
            "textColor": "#000000",
            "backgroundColor": color           
        }

    try: 
        label = gmail.users().labels().create(
            userId='me',
            body=label_body
        ).execute()
    except HttpError as error:
        print(f'An error occurred: {error} label_name: {label_name}')
        return

    if description:
        add_label_descriptor(label_name, label['id'], description)
        load_map_of_labels()
        utils.update_json('config.json', "user_labels", label_descriptors)

    return label['id']

def get_backlogged_emails() -> list[Email]:
    global gmail

    if id := utils.get_json_field('config.json', 'history_id'):
        new_emails = _retrieve_new_emails(gmail, id)
        utils.update_json('config.json', 'history_id', gmail.users().getProfile(userId='me').execute()['historyId'])
        return new_emails
        
    return []

def retrieve_email_from_id(id):
    return _retrieve_email_from_id(gmail, id)
def _retrieve_email_from_id(gmail: Resource, id) -> Email:

    msg_data = gmail.users().messages().get(userId='me', id=id, format='full').execute()

    payload = msg_data.get('payload', {})
    headers = payload.get('headers', [])

    header_dict = {h['name']: h['value'] for h in headers}

    email_args = {
        "sender": header_dict['From'],
        "recipients": header_dict['To'].split(',') if 'To' in header_dict else [],
        "sentOn": parser.parse(header_dict['Date']).date() if 'Date' in header_dict else date.today(),
        "subject": header_dict.get('Subject', None),
        "email_id": id,
        "label_names": [ids_to_names[l_id] for l_id in msg_data.get('labelIds', []) if l_id in ids_to_names],
        "text": "null"
    }
    email = Email.model_validate(email_args)
    
    if s := extract_email_text(payload):
        body = strip_whitespace(
          strip_html(
            strip_urls(
              strip_invisible_chars(
                unicodedata.normalize("NFKC",s)
              )
            )
          )
        ).strip()
        email.text = re.sub(r'\n+', '\n', body)
    else:
        email.text = "null"
    
    return email

def _retrieve_new_emails(worker_gmail: Resource, history_id) -> list[Email]:
    new_emails = []
    page_token = None

    try:
        while True:
            
            results = worker_gmail.users().history().list(
                userId='me',
                startHistoryId=history_id,
                historyTypes=['messageAdded'],
                pageToken=page_token
            ).execute()

            history = results.get('history', {})

            new_email_ids = []
            for update in history:
                for message in update.get('messagesAdded', []):
                    new_email_ids.append(message['message']['id'])
            
            for id in new_email_ids:
                new_emails.append(_retrieve_email_from_id(worker_gmail, id))

            page_token = results.get('nextPageToken')
            if not page_token:
                break
    except Exception as e:
        print(e)

    return new_emails



def _check_for_new_email(worker_gmail: Resource, func: Callable, *args):
    
    latest_id = worker_gmail.users().getProfile(userId='me').execute()['historyId']
    utils.update_json('config.json', 'history_id', latest_id)

    while True:
        time.sleep(5.)
        new_id = worker_gmail.users().getProfile(userId='me').execute()['historyId']
        if latest_id != new_id:
            if new_emails := _retrieve_new_emails(worker_gmail, latest_id):
                func(new_emails, *args)
            latest_id = new_id
            utils.update_json('config.json', 'history_id', latest_id)
            
        
def start_email_checking(creds, func: Callable, *args):
    worker_gmail = build('gmail', 'v1', credentials=creds)
    thread = threading.Thread(target=_check_for_new_email, daemon=True, args=(worker_gmail, func, *args))
    thread.start()

    
def init_gmail(creds):
    global gmail, user_email, map_of_labels

    gmail = build('gmail', 'v1', credentials=creds)
    user_email = gmail.users().getProfile(userId='me').execute()['emailAddress']
    check_label_ids()



def keyword_query_inbox(keywords: str, subject: str = None, start: date = None, end: date = None, 
                sender: str = None) -> list[Email]:
    print(keywords)
    subject_query = '' if subject is None else f'subject:{subject} '
    start_query = '' if start is None else f'after:{start.strftime("%Y/%m/%d")} '
    end_query = '' if end is None else f'before:{end.strftime("%Y/%m/%d")} '
    sender_query = '' if sender is None else f'from:{sender} '
    query = keywords + " " + subject_query + start_query + end_query  + sender_query
    
    results = gmail.users().messages().list(userId='me', q=query, maxResults=10).execute()

    messages = results.get('messages', [])
    msgs = [msg['id'] for msg in messages]

    query_results = []

    for msg in msgs:
        email = _retrieve_email_from_id(gmail, msg)
        query_results.append(email)

    return query_results
   
    

def semantically_query_inbox(query: str, start: date = None, end: date = None) -> dict:

    email_ids = semantics.query_index(query, 5, start, end)
    seen = set()
    i = 0

    while i < len(email_ids):
        if email_ids[i] in seen:
            del email_ids[i]
        else:
            seen.add(email_ids[i])
            i += 1


    emails = [_retrieve_email_from_id(gmail, e_id).to_dict() for e_id in email_ids]

    
    return json.dumps({
        "status": "success",
        "result": "fetched query matching emails",
        "emails": emails
    }, indent=2)

def strip_urls(text: str) -> str:
    return re.sub(r'https?://[^\s]+',replace_helper, text)

def strip_html(html: str) -> str:
    newline_tags = r'</?(p|div|br|li|ul|ol|tr|h[1-6])[^>]*>'
    text = re.sub(newline_tags, '\n', html, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(&zwnj;)+', '', text)
    return text

def replace_helper(match: re.Match) -> str:
    url = match.group()
    try:
        domain = urlparse(url).netloc
    except Exception as error:
        return ""
    return domain.removeprefix("www.")

def extract_email_text(part):
    if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
        return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='replace')
    elif 'parts' in part:
        for sub_part in part['parts']:
            body = extract_email_text(sub_part)
            if body:
                return body
    return None

def strip_whitespace(text: str) -> str:
    return re.sub(r'\n +', "\n", re.sub(r'(?:\r\n)+', "\n", re.sub(r'( |\t)+', " ", text)))

def strip_invisible_chars(text: str) -> str:
    # Matches zero-width joiner, zero-width non-joiner, non-breaking space, etc.
    invisible_chars = [
        '\u034f',  # combining grapheme joiner
        '\u200b',  # zero-width space
        '\u200c',  # zero-width non-joiner
        '\u200d',  # zero-width joiner
        '\u2060',  # word joiner
        '\ufeff',  # BOM
        '\u00a0',  # non-breaking space
        '\u2019',
        '\u2013'
    ]
    pattern = "[" + "".join(invisible_chars) + "]"
    return re.sub(pattern, "", text)

init_gmail(utils.creds)
  



        