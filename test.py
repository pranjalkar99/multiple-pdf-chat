from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from typing import List

app = FastAPI()
load_dotenv()
conversation_chain = {}
chat_history = {}
vectorstore = {}


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf.file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_user_input(user_question, user_id, session_id):
    chain = conversation_chain.get(user_id, {}).get(session_id)
    if chain is None:
        return None, None

    response = chain({'question': user_question})
    history = chat_history.get(user_id, {}).get(session_id)
    if history is None:
        history = []

    history.extend(response['chat_history'])
    answer = response['answer']

    return answer, history


@app.post("/upload")
async def upload_pdfs(user_id: str, session_id: str, files: List[UploadFile] = File(...)):
    global vectorstore, conversation_chain, chat_history

    raw_text = get_pdf_text(files)
    text_chunks = get_text_chunks(raw_text)

    if user_id not in vectorstore:
        vectorstore[user_id] = {}

    vectorstore[user_id][session_id] = get_vectorstore(text_chunks)
    print(vectorstore)

    if user_id not in conversation_chain:
        conversation_chain[user_id] = {}

    if session_id not in conversation_chain[user_id]:
        vectorstore_instance = vectorstore[user_id][session_id]
        conversation_chain[user_id][session_id] = get_conversation_chain(vectorstore_instance)
    print(conversation_chain)

    if user_id not in chat_history:
        chat_history[user_id] = {}

    if session_id not in chat_history[user_id]:
        chat_history[user_id][session_id] = []
    print(chat_history)

    return {"message": "PDFs uploaded and processed successfully"}


@app.post("/chat")
async def chat_with_pdf(user_id: str, session_id: str, question: str):
    answer, user_chat_history = handle_user_input(question, user_id, session_id)
    return {"answer": answer}#, "chat_history": user_chat_history}

@app.get("/clear")
async def clear_db(user_id: str, session_id: str):
    del chat_history[user_id]
    del vectorstore[user_id]
    del conversation_chain[user_id]
    return JSONResponse({"status":"Removed old configuration.. Upload again"})