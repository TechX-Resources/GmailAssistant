from datetime import date
from datetime import timedelta
import semantics
import gmail_tools
import gmail_tools
from data_schemas import Email

def generate_index():
    today_time = date.today()
    timestep = timedelta(days=5)
    curr_time = today_time - timedelta(days=365)


    while curr_time < today_time:
        print(curr_time)
        emails = gmail_tools.keyword_query_inbox('', start=curr_time, end=curr_time + timestep)
        print(len(emails))
        

        semantics.add_embeddings(emails)
        curr_time += timestep

    
    semantics.save_index()





