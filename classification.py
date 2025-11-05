import json
from pydantic import BaseModel, Field
from data_schemas import Email
from langchain_google_genai import ChatGoogleGenerativeAI
from enum import Enum
import os
import gmail_tools
import utils

labelMap = {}
UserLabelEnum = None


os.environ["GOOGLE_API_KEY"] = utils.get_json_field('config.json', 'gemini_key')


enum_vals = { label['name']: label['name'] for label in gmail_tools.label_descriptors if label['name'] != 'scheduled' }
enum_vals.update({"irrelevant": "irrelevant"})

label_map = { label['name']: label['description'] for label in gmail_tools.label_descriptors if label['name'] != 'scheduled' }
label_map.update({ "irrelevant": "Emails that don't have anyting to do with any of these categories." })

UserLabelEnum = Enum("UserLabelEnum", enum_vals)

print(UserLabelEnum)

class UserLabelClassification(BaseModel):
    classification: UserLabelEnum = Field(...,
        description=f"Classifies the content of the email.\n\n Possible labels:\n {"\n".join([key + ": " + label_map[key] for key in label_map])}"
    )
    summary: str = Field(..., description="Short blurb about the contents of the email.")


classifier = ChatGoogleGenerativeAI(model='gemini-2.5-flash').with_structured_output(UserLabelClassification)

def classify(emails: list[Email]):
    values = []
    
    for e in emails:
        resp = classifier.invoke(make_prompt(e.text))
        if resp.classification.value != "irrelevant":
            gmail_tools.add_email_label(e, resp.classification.value)
        
        values.append(resp.classification.value)

    return values

def make_prompt(email):
    return f"""You are an email inbox assistant. Classify and summarize this email in a structured output:
{email}"""

