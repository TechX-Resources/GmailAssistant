## Gmail Assistant

This project implements agentic use of Large Language Models, semantic embedding databases, and KNN to create a proactive gmail and calendar chatbot that notifies the user of events and incoming emails. It consists of 3 agents: the `chatbot` whose purpose is to talk to the user and relay tasks to the other agents, the `gmail_agent` whose purpose is to search for emails and manage email labeling, and the `calendar_agent` that manages the user's google calendar.

### Features
1. Can preprocess and vectorize large volumes of emails in minutes. Process is fully automated and the database can be queried against after running `generate_semantic_index` for your desired time interval
2. Can query against the database using linear search. Accuracy of search results is often higher than with the keyword search Gmail natively supports
3. Fully functional chatbot. Implemented an end-to-end langgraph chatbot interface with tool routing to increase accuracy
4. Calendar management. The calendar agent has access to your calendar and can create, reschedule, and cancel events and tasks. Can also attach reminders to tasks that will email or notify you of the task at a designated time.
5. Gmail management. The gmail agent has access to your gmail inbox and can search for messages by keyword or through semantic meaning via the vector database. Can also manage labels attached to emails.

### Previews
<img width="600" height="380" alt="Screenshot 2025-10-01 at 1 04 32 AM" src="https://github.com/user-attachments/assets/a41d4cae-0494-4501-a26e-5cea76a240fa" /><img width="300" height="300" alt="Screenshot 2025-10-01 at 10 34 35 PM" src="https://github.com/user-attachments/assets/634cc272-1202-4d58-bfd0-ed54620b551a" />


<img width="600" height="285" alt="Screenshot 2025-10-01 at 7 49 17 PM" src="https://github.com/user-attachments/assets/818eca47-96f7-4046-996a-46e6c377e151" /><img width="300" height="175" alt="Screenshot 2025-10-01 at 7 50 11 PM" src="https://github.com/user-attachments/assets/8eba2cfe-e4d3-42d6-9ffa-6d39dca2fe2a" />



### Problems to Solve
1. Gmail inboxes don't have categories for emails that might require a reply or action, and sometimes you may forget to reply to an email or perform an action specified in an email.
2. Many Gmail inboxes aren't checked for long periods of times, or receive large volumes of emails in short amounts of times that are difficult to navigate.
3. When receiving an email about an upcoming event, it's sometimes cumbersome to have to switch between Gmail and google calendar to check availability.

### Solution
My solution is to implement a gmail assistant that exists as a popup box in the corner of the screen. Through this interface, the user will be able to query their inbox for information, preparing a brief of unread emails, check availability on a certain day or time, schedule reminders in google calendar for replies or actions, as well as events. The assistant will also be proactive, automatically categorizing emails into different categories based on content and providing useful suggestions to the user based on that categorization.

#### Labeling
1. The assistant will first categorize emails into important and unimportant. Personal emails from other individuals will be prioritized and promotional and social emails will be marked as unimportant by default. The user can also provide input to what emails should be marked as important or unimportant, such as, 'all emails from linkedin are important' or 'all emails pertaining to social media are important'.
2. The assistant will then be given a list of predecided labels for to categorize emails into the labels 'needs reply', 'action required', 'needs scheduling', and 'miscellaneous information'. It will then retroactively apply these labels to existing emails in the inbox. Incoming emails will automatically be categorized into the correct labels. The Assistant will them ask the user what action they might want to take with this email, i.e. 'do you want to schedule a reminder to reply to this email' or 'do you want me to give you your availability on the day of this event?'.

Possible features: 
1. More specific categorizing based off of user input, e.g. 'organize the subscriptions label by company', 'put all emails older than a year in a 'to delete' label'.

#### Summarizing
The assistant will prepare briefs after the inbox between certain dates. The user will also be able to query emails to summarize, e.g. 'summarize my last conversation with joe bradford'.

#### Asking the Inbox
The user will be able to ask information that might be contained in the inbox similar to a google search. The user will be able to ask questions such as 'when is my doctor's appointment'.

#### Calendar and Availability
The assistant will also be connected to the user's google calendar and will be able to check availability. The user will be able to ask for their availability, and if there are possible conflicts the assistant will list them. The assistant will also be able to create and delete events on user request.

### Implementation
#### Tech Stack/ Modules Needed
1. Google API services and authorization - To handle logging into services and retrieving emails and google calendar evemts.
2. Google Gemini LLM - To handle tasks such as creating labels, filters, and summarizing text.
3. LangChain and LangGraph - To handle the thinking aspects of the assistant, i.e. figuring out which functions to call based on user request.
4. Pydantic - For JSON validation of event objects and descriptions of tool arguments.

#### Labeling Implementation Steps
1. Create a gmail tools module that contains functions that can retrieve emails from google APIs and organize them into Email objects.
2. Engineer a prompt to feed to the LLM that will categorize these emails into the labels.
3. Create a function that categorizes already received emails as well as incoming emails

**We will finish this feature and then focus on implementation of the others later**
