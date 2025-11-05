
import threading
import semantics
import classification
import gmail_tools
import utils
import chatbot
from pynput import mouse
from datetime import date
from data_schemas import Email

notifier = threading.Condition()
agent_relevant_categories = set(['needs_action', 'to_schedule'])

agent_relevant_emails = []
end = False



def process_emails(emails: list[Email]):
    semantics.add_embeddings(emails)
    classifications = classification.classify(emails)
    for i in range(len(emails)):
        if classifications[i] in agent_relevant_categories:
            agent_relevant_emails.append(emails[i])




def mouse_listener():

    def on_click(x, y, button, pressed):
        global end
        if pressed:
            if button == mouse.Button.right:
                end = True

            with notifier:
                notifier.notify()

    with mouse.Listener(on_click=on_click) as listener:
        listener.join()


process_emails(gmail_tools.get_backlogged_emails())

gmail_tools.start_email_checking(utils.creds, process_emails)
threading.Thread(target=mouse_listener, daemon=True).start()


while True:
    with notifier:
        notifier.wait()
        
    if end: break

    if agent_relevant_emails:
        for event in chatbot.start_new_email_agent(agent_relevant_emails.pop(0)):
            pass
    else:

        for event in chatbot.start_new_conversation_agent():
            pass