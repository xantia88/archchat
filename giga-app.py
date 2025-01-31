from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # load environment variables
    load_dotenv()

    # initialize LLM object
    llm = GigaChat(
        credentials=os.environ["auth_key"],
        scope="GIGACHAT_API_PERS",
        model="GigaChat",
        streaming=False,
        verify_ssl_certs=False,
    )

    # prepare request
    messages = [
        SystemMessage("Переведи следующее сообщение с русского на английский"),
        HumanMessage("привет! сегодня хороший день"),
    ]

    # request
    resp = llm.invoke(messages)

    # response
    print(resp)
