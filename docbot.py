import streamlit as st
import fitz
import os
import docx
import csv
from streamlit_chat import message
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFaceHub
from sentence_transformers import SentenceTransformer

# Initialize Hugging face api key 
Hugging_Face_Api_key = st.secrets['HUGGING_FACE_API_KEY']
def main():
    # Setup the page configuration
    st.set_page_config(page_title='SuperDocBot')
    st.header('Super Document GPT')

    # Initialize session state Variable 
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None
    if 'processComplete' not in st.session_state:
        st.session_state.processComplete = None
        
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload Your Files" , type = ['pdf'] , accept_multiple_files=True)
        # Create a button
        process = st.button("Process")
    if process:
        if not Hugging_Face_Api_key:
            st.info("Please Add your Hugging Face key")
            st.stop()
        
        files_text = get_files_text(uploaded_files)
        st.write("File loaded..")
        text_chunk = get_text_chunks(files_text)
        st.write("File Chunks Created...")
        vectorStore = get_vector_store(text_chunk)
        st.write("Vector store created...")

        st.session_state.conversation = get_conversation_chain(vectorStore, Hugging_Face_Api_key)

        st.session_state.processComplete = True
    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Questions related to your document")
        if user_question:
            handel_user_input(user_question)
                
                
def get_files_text(upload_file):
    text = ''
    for uploadfile in upload_file:
        split_up = os.path.splitext(uploadfile.name)
        file_extension = split_up[1]
        
        if file_extension == '.pdf':
            text += get_pdf_text(uploadfile)
        elif file_extension == '.docx':
            text += get_docx_text(uploadfile)
        
        else:
            text += get_csv_text(uploadfile)
    return text

def get_pdf_text(pdf):
    document = fitz.open(stream= pdf.read(), filetype= 'pdf')
    text = ''
    for page in document:
        text += page.get_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []

    for docpara in doc.paragraphs:
        allText.append(docpara.text)
        text = ''.join(allText)
    return text

def get_csv_text(file):
    
    # Read CSV 
    decoded_file = io.TextIOWrapper(file , encoding='utf-8')
    csv = csv.reader(decoded_file)
    
    for row in csv:
        # Separating each row into a single line text, separating columns with a space or comma
        text = ''.join(row) + '\n'
    return text

def get_text_chunks(text):
    textsplit = CharacterTextSplitter(
        separator='\n',
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunk = textsplit.split_text(text)
    return  chunk

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")

    Knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return Knowledge_base

def get_conversation_chain(vector_store, HuggingKey):
    llm = HuggingFaceHub(repo_id= "google/flan-t5-large", huggingfacehub_api_token= Hugging_Face_Api_key)
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        memory = memory,
        retriever = vector_store.as_retriever()
    )
    return conversation_chain

def handel_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, key= str(i), is_user=True)
            else:
                message(messages.content, key= str(i))

if __name__ == "__main__":
    main()