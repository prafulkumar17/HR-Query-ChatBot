from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def upload_htmls():
    """
    This function does the following:
    -> Reads recursively through thhe given golder hr-policies(without current folder)
    -> Loads the Pages(Docs)
    -> Loaded documents are split into chunks using Splitter
    -> These chunks are converted into Language Embeddings and loaded as vector into a local FAISS vector db
    """
    loader = DirectoryLoader(path = "C:/Personal_1/Praful/RAG_Udemy/langchain-docs")
    documents = loader.load()
    print(f"{len(documents)} Pages Loaded")

    #Split laoded documents into chunks using CharacterTextASplitter
    text_splitter  = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators=["\n\n","\n"," ",""]
    )
    split_doc = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(split_doc)} Documents...")

    print(split_doc[0].metadata)
    #upload chunks as vector embeddings into FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(split_doc, embeddings)
    db.save_local("faiss_index")

def faiss_query():
    """
    This function does the following:
    -> Load the local FAISS db
    -> Trigger a Semantic Similarity Search using a query
    -> This retieves semantically matching vectors from the db
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    query = "Explain the Candidate Onboarding process"
    docs = new_db.similarity_search(query)

    # Print all the extracted Vectors from the above Query
    for doc in docs:
        print("##--- Page ---##")
        print(doc.metadata['source'])
        print("##--- Page ---##")
        print(doc.page_content)

if __name__ == "__main__":
    #The below code 'uploaded_htmls()' is executed only once and then commmented as
    #the vector db is now built and ready for your further experiments
    # upload_htmls()
    #The below function is experimental to trigger a semantic search on the vector db
    faiss_query()
