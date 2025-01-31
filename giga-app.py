from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def execute(llm, messages):
    response = llm.invoke(messages)
    print(response)


def create_system_message(command, context):
    if context:
        return "{}.{}".format(context, command)
    else:
        return command


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
    context = Path("data/context.txt").read_text()
    system_message = "{}.{}".format(
        context, "Замени в описании системы значения параметров на их текстовые описания")

    messages = [
        SystemMessage(system_message),
        HumanMessage(
            "{'type': 'system', 'name':'моя система', 'class': 'управление проектами', 'level': 'high', 'location': 'внешнее'}"),
    ]

    # execute request
    execute(llm, messages)
