import streamlit as st
import cohere  
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from concurrent.futures import ThreadPoolExecutor

# Access the API key value
#api_key = st.secrets['YOUR_SECRET']
api_key = '4jdEqGb3coPXaw7M8mbEaTvKMdYu5vJsa0G3MHbL'

co = cohere.Client(api_key) 

# add title
st.title("Customer Service Bot")
# add a subtitle
st.subheader("Cohere hackathon product demo made by\n Dvir Zagury, Ellie Lastname, Joschua Lastname.")


# Load the search index
search_index = AnnoyIndex(f=4096, metric='angular')
search_index.load('search_index.ann')

# load the csv file called cohere_final.csv
df = pd.read_csv('cohere_text_final.csv')

def search(query, n_results, df, search_index, co):
    # Get the query's embedding
    query_embed = co.embed(texts=[query],
                    model="large",
                    truncate="LEFT").embeddings
    
    # Get the nearest neighbors and similarity score for the query and the embeddings, 
    # append it to the dataframe
    nearest_neighbors = search_index.get_nns_by_vector(
        query_embed[0], 
        n_results, 
        include_distances=True)
    # filter the dataframe to include the nearest neighbors using the index
    df = df[df.index.isin(nearest_neighbors[0])]
    index_similarity_df = pd.DataFrame({'similarity':nearest_neighbors[1]}, index=nearest_neighbors[0])
    df = df.join(index_similarity_df,) # Match similarities based on indexes
    df = df.sort_values(by='similarity', ascending=False)
    return df


# define a function to generate an answer
def gen_answer(q, para): 
    response = co.generate( 
        model='command-light', 
        prompt=f'''Paragraph:{para}\n\n
                Answer the question using this paragraph.\n\n
                Question: {q}\nAnswer:''', 
        max_tokens=100, 
        temperature=0.4)
    return response.generations[0].text


def find_intent(query):
    # Examples for each category
    examples_intent = [
        # Technical Support
        Example("I can't log into my account.", "Technical Support"),
        Example("The application crashes when I try to upload a file.", "Technical Support"),
        Example("I'm experiencing slow response times from the server.", "Technical Support"),

        # Pricing and Billing
        Example("Can you confirm if my payment went through?", "Pricing and Billing"),
        Example("I've been overcharged for last month's service.", "Pricing and Billing"),
        Example("How do I upgrade my current plan?", "Pricing and Billing"),

        # Product Information
        Example("Can your product support multi-language inputs?", "Product Information"),
        Example("Does your AI support image recognition?", "Product Information"),
        Example("Is your software compatible with macOS?", "Product Information"),

        # Development Support
        Example("How can I integrate your API with my existing systems?", "Development Support"),
        Example("I need help with a script using your AI model.", "Development Support"),
        Example("What are the limitations of using the free version of your platform?", "Development Support"),

        # Careers
        Example("Are there any current openings for data scientists?", "Careers"),
        Example("What's the status of my job application submitted last week?", "Careers"),
        Example("Do you offer internships for undergraduates?", "Careers"),

        # Partnership and Collaboration
        Example("We're interested in a strategic partnership with your company.", "Partnership and Collaboration"),
        Example("Would you be interested in a joint research initiative?", "Partnership and Collaboration"),
        Example("We are seeking sponsors for our tech conference.", "Partnership and Collaboration"),

        # Feedback and Suggestions
        Example("I have some suggestions to improve the user interface.", "Feedback and Suggestions"),
        Example("Consider adding a dark mode feature to your app.", "Feedback and Suggestions"),
        Example("I am not satisfied with the customer service I received.", "Feedback and Suggestions"),

        # Marketing
        Example("How can we advertise our products on your platform?", "Marketing"),
        Example("Can you provide details about your upcoming webinar?", "Marketing"),
        Example("I'm a journalist looking for a press contact at your company.", "Marketing")
    ]

    response_intent = co.classify(
    model='embed-english-v2.0',
    inputs=[query],
    examples=examples_intent
    )

    return response_intent.classifications[0].predictions[0], response_intent.classifications[0].confidences[0]
    


def find_urgency():
    ...





def gen_better_answer(ques, ans): 
    response = co.generate( 
        model='command-light', 
        prompt=f'''Answers:{ans}\n\n
                Question: {ques}\n\n
                Generate a new answer that uses the best answers 
                and makes reference to the question.''', 
        max_tokens=100, 
        temperature=0.4)
    return response.generations[0].text




############# Algorithm structure: #############

if find_intent(query)[0] == "Technical Support":
    prompt = f"You recieved a customer support query with intent identified as technical support. Find an answer: {query} "


























def display(query, results):
    # 1. Run co.generate functions to generate answers

    # for each row in the dataframe, generate an answer concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        results['answer'] = list(executor.map(gen_answer, 
                                              [query]*len(results), 
                                              results['text_chunk']))
    answers = results['answer'].tolist()
    # run the function to generate a better answer
    
    # BETTER ANSWER
    answ = gen_better_answer(query, answers)
    
    # 2. Code to display the resuls in a user-friendly format

    st.subheader(query)
    st.write(answ)
    # add a spacer
    st.write('')
    st.subheader("Here's XXX:")


# add the if statements to run the search function when the user clicks the buttons

query = st.text_input('Ask our cutsomer service bot a question.')
# write some examples to help the user

st.markdown('''Try some of these examples: 
- Example1
- Example2
- Example3''')

if st.button('Search'):
    results = search(query, 3, df, search_index, co)
    display(query, results)
