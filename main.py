from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_together import Together
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

load_dotenv()

def get_qa_chain():
    loader = PyPDFLoader("amara_grove_brochure.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding=embeddings)

    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.5
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), return_source_documents=True)
    return qa_chain