from data_schemas import Email
from datetime import date
import numpy as np

import faiss
import requests
import os
import google.auth

from google.auth.transport.requests import Request
import pandas as pd
import utils

# Get credentials
creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
creds.refresh(Request())

EMBEDDING_SIZE = 3072

url = "https://us-central1-aiplatform.googleapis.com/v1/projects/starry-diode-464720-n0/locations/us-central1/publishers/google/models/gemini-embedding-001:predict"

headers = {
    "Authorization": f"Bearer {creds.token}",
    "Content-Type": "application/json"
}


index = faiss.IndexFlatIP(EMBEDDING_SIZE)

data = utils.load_data()

def parse_embeddings_json(response: requests.Response):
    obj = response.json()
    embeddings_list = []

    for prediction in obj['predictions']:
        embedding = prediction['embeddings']['values']
        embeddings_list.append(embedding)

    return embeddings_list


def add_embeddings(emails: list[Email]):
    global index, data

    for i in range(0, len(emails), 10):
        embedding_map = split_texts(emails[i:i+10])
        

        

        response = requests.post(url, headers=headers, json= { "instances": [{"content": text} for text in embedding_map[1]] })

        data = pd.concat((
            data, 
            pd.DataFrame(
                np.concat((embedding_map[0], parse_embeddings_json(response)), axis=1), 
                columns=['Date', 'Email_IDs'] + [f'Embedding_{i}' for i in range(EMBEDDING_SIZE)]
            )
        ))
        
    
def query_index(query: str, k: int, start: date = None, end: date = None) -> list[str]:
    global index, data


    if not start:
        start = data.index[0].date()
    
    if not end:
        end = data.index[len(data)-1].date()

    relevant_data = data[start:end]

    email_ids = relevant_data['Email_IDs'].to_numpy()
    embeddings = relevant_data[[f'Embedding_{i}' for i in range(EMBEDDING_SIZE)]].to_numpy()


    response = requests.post(url, headers=headers, json= { "instances": [{"content": query}] })
    query_embedding = parse_embeddings_json(response)[0]
    
    index.add(np.ascontiguousarray(embeddings))
    D, I = index.search(np.ascontiguousarray(
        np.array(query_embedding, dtype='float32').reshape(1, EMBEDDING_SIZE)
    ), k)

    index.reset()
    return [email_ids[i] for i in I[0] if i != -1]

def split_texts(emails: list[Email]):

    embedding_map = ([], [])
    for email in emails:
        email_txt = email.model_dump_json(exclude={'sentOn', 'email_id'})
        for i in range(0, len(email.text), 900):
            embedding_map[0].append([email.sentOn, email.email_id])
            embedding_map[1].append(email_txt[i: i + 1000])

    return embedding_map


def save_index():
    utils.save_data(data)

