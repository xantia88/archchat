from langchain_gigachat.chat_models import GigaChat
import os
from dotenv import load_dotenv
import warnings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")


def get_response(llm, messages):
    response = llm.invoke(messages)
    parser = StrOutputParser()
    content = parser.invoke(response)
    return AIMessage(content)


if __name__ == "__main__":

    load_dotenv()

    llm = GigaChat(
        credentials=os.environ["auth_key"],
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Max",
        verify_ssl_certs=False,
    )

    texts = [
        "привет, меня зовут Алекс",
        "как меня зовут?"
    ]
    messages = []
    for text in texts:
        m = HumanMessage(text)
        messages.append(m)
        r = get_response(llm, messages)
        messages.append(r)
        print(m)
        print(r)
