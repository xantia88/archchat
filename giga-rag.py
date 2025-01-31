import os
from dotenv import load_dotenv
import warnings
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_gigachat.chat_models import GigaChat
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

warnings.filterwarnings("ignore")
load_dotenv()

llm = GigaChat(
    credentials=os.environ["auth_key"],
    scope="GIGACHAT_API_PERS",
    model="GigaChat",
    streaming=False,
    verify_ssl_certs=False
)

loader = TextLoader("data/systems.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings_function = SentenceTransformerEmbeddings(
    model_name='all-MiniLM-L6-V2')
db = Chroma.from_documents(docs, embeddings_function)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
ans = qa_chain({"query": "сколько внешних систем?"})
print(ans)
