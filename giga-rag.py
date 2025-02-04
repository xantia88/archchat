from os.path import isfile, join
import os
from os import listdir
from pathlib import Path
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document

warnings.filterwarnings("ignore")


def translate(llm, filepath):

    content = Path("config/content.txt").read_text()
    context = f"Используй следующий контекст: {content}"
    command = "Cоставь краткое текстовое описание"
    data = Path(filepath).read_text()

    messages = [
        SystemMessage(f"{context}. {command}"),
        HumanMessage(data)
    ]

    response = llm.invoke(messages)
    parser = StrOutputParser()
    text = parser.invoke(response)
    return text


if __name__ == "__main__":

    # create prompt
    question = ("Какие поля в Р11 обязательны к заполнению?"
                ""
                ""
                "")

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
    path = "data"
    data = []
    files = [file for file in listdir(path) if isfile(join(path, file))]
    for file in files:
        ext = file.split(".")[1].lower()
        filepath = join(path, file)
        if ext == "txt":
            loader = TextLoader(filepath)
            doc = loader.load()
            data.extend(doc)
        elif ext in ["json", "yaml"]:
            text = translate(llm, filepath)
            print("[TRANSLATE]", text)
            doc = [Document(page_content=text)]
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
    print("[QUESTION]", question)
    response = qa_chain({"query": question})

    # response
    print("[ANSWER]", response["result"])
