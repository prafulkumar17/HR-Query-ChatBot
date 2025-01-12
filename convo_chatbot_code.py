import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()

def query(question, chat_history):
    """
    This Function does the following:
    -> Receives two parameters - 'question' - a string and 'chat_history' - a Python List of tuples containing 
        accumulated question-answer pairs
    -> Load the local FAISS database where the entire website is storedas Embedding vectors
    -> Create a ConversationalBufferMemory object with 'chat_history'
    -> Create a ConversationalRetrievalChain object with FAISS db as the Retiever (LLM lets create retriever objects against data stores)
    -> Invoke the retriever object with thhe query and chat history
    -> Return the response
    """

    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro", temperature=0, api_key=os.getenv("GOOGLE_API_KEY"))
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    #Initialisze a ConversationalRetrievalChain
    query = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = new_db.as_retriever(),
        return_source_documents = True)
    #Invoke the Chain with
    return query({"question":question, "chat_history":chat_history})


def show_ui():
    """
    This fucntion does the following:
    -> Implements the Streamlit UI
    -> Implements two session_state variables - 'messages' - to contain the accumulating question and answers to be
        displayed on the UI and  'chat_hhistory' - the accumulating question answer pairs as a List of tuples to be served
        to the retriever object as chat_history
    -> For each user query, the responcse is obtained by invoking the 'query' function and the chat histories are built up
    """

    st.title("Human Resources Chatbot")
    # st.image("CGAQ9153.JPG")
    st.subheader("Enter the HR Query")
    #Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    #Accept an Input
    #chat_input is used for making a text box for input
    # := is a walrus operator -- used for assigning a var in a larger expression
    if prompt := st.chat_input("Enter your HR policy related Query:"):
        #Invoke the function with the retriever with chat history and display responses in chat container in question-answer pair
        #spinner used to show the loading text
        with st.spinner("Working on your query..."):
            response = query(question=prompt, chat_history=st.session_state.chat_history)
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])

            #Append user message to chat history
            st.session_state.messages.append({"role":"user", "content":prompt})
            st.session_state.messages.append({"role":"assistant", "content": response["answer"]})
            st.session_state.chat_history.extend([(prompt, response["answer"])])

if __name__ == "__main__":
    show_ui();


