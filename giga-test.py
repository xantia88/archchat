import os
from dotenv import load_dotenv
from langchain_gigachat.chat_models import GigaChat

load_dotenv()

model = GigaChat(
    credentials=os.environ["auth_key"],
    scope="GIGACHAT_API_PERS",
    model="GigaChat",
    streaming=False,
    verify_ssl_certs=False,
)

resp = model.get_models()
print(resp)
