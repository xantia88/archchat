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
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document

warnings.filterwarnings("ignore")


def request(llm, context, question):

    messages = [
        SystemMessage(context),
        HumanMessage(question)
    ]

    response = llm.invoke(messages)
    parser = StrOutputParser()
    text = parser.invoke(response)
    return text


def translate(llm, content_file, filepath):
    command = "Cоставь краткое текстовое описание системы"
    content = Path(content_file).read_text()
    context = f"Используй следующий контекст: {content}"
    context = f"{context}. {command}"
    data = Path(filepath).read_text()
    return request(llm, context, data)


def load_documents(path, content_file):
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
            text = translate(llm, content_file, filepath)
            print(f"[TRANSLATE {filepath}]", text)
            doc = [Document(page_content=text)]
            data.extend(doc)
    return data


if __name__ == "__main__":

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
    documents = load_documents("documents", "config/terms.txt")
    n = len(documents)
    print(n, "documents loaded")
    filename = "documents/data.txt"
    with open(filename, "w") as file:
        for document in documents:
            file.write(document.page_content)
            file.write("\n")

    print("data saved to", filename)
