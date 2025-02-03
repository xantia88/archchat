from os.path import isfile, join
import os
from os import listdir
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    question = "сколько внешних систем?"
    # question = "сколько клиентов у компании?"
    path = "data"

    # load environment variables
    load_dotenv()

    # initialize LLM object
    llm = GigaChat(
        credentials=os.environ["auth_key"],
        scope="GIGACHAT_API_PERS",
        model="GigaChat",
        streaming=False,
        verify_ssl_certs=False
    )

    # load content
    data = []
    files = [file for file in listdir(path) if isfile(join(path, file))]
    for file in files:
        loader = TextLoader(join(path, file))
        doc = loader.load()
        data.extend(doc)

    # split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(data)

    # create embeddings and put them into in-memory vector storage
    db = Chroma.from_documents(documents, SentenceTransformerEmbeddings(
        model_name='all-MiniLM-L6-V2'))

    # prepare LLM to process request
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

    # request
    response = qa_chain({"query": question})

    # response
    print(response["result"])
