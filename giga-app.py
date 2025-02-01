from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def execute(llm, messages):
    parser = StrOutputParser()
    response = llm.invoke(messages)
    s = parser.invoke(response)
    print(s)


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

    # prepare simple request
    messages = [
        SystemMessage("Переведи сообщение с русского на английский"),
        HumanMessage("привет! сегодня хороший день, тысяча чертей"),
    ]

    # execute request
    execute(llm, messages)

    # prepare request with a context
    context = Path("data/app/context.txt").read_text()
    command = "Замени параметры на их текстовое описание, и составь краткое текстовое описание системы"
    system_message = "{}.{}".format(context, command)
    user_message = Path("data/app/systems.json").read_text()

    messages = [
        SystemMessage(system_message),
        HumanMessage(user_message),
    ]

    # execute request
    execute(llm, messages)
