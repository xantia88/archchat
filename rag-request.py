import os
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # create prompt
    question = "сколько внешних систем?"

    # load environment variables
    load_dotenv()

    # initialize LLM object
    llm = GigaChat(
        credentials=os.environ["auth_key"],
        scope="GIGACHAT_API_PERS",
        model="GigaChat",
        streaming=False,
        verify_ssl_certs=False)

    # load content
    loader = TextLoader("documents/data.txt")
    data = loader.load()

    # split text into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)

    # create embeddings and put them into in-memory vector storage
    db = Chroma.from_documents(documents, SentenceTransformerEmbeddings(
        model_name='all-MiniLM-L6-V2'))

    # request / response
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    print("[QUESTION]", question)
    response = qa_chain({"query": question})
    print("[ANSWER]", response["result"])
