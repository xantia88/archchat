from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings("ignore")


def request(llm, messages):
    response = llm.invoke(messages)
    print(response)


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
        SystemMessage("Переведи сообщение с русского на английский"),
        HumanMessage("привет! сегодня хороший день, тысяча чертей"),
    ]

    # request 1
    request(llm, messages)

    messages = [
        SystemMessage("Расположение системы описывается параметром location, расположение может быть либо внутренним либо внешним. Система описывается как объект с типом system. Уровень критичности системы описывается параметром level.Класс системы описывается параметром class. Класс системы может принимать одно из следующих значений: управление проектами, бухгалтерия, коммуницация. Замени в описании системы значения параметров на их текстовые описания."),
        HumanMessage(
            "{'type': 'system', 'name':'моя система', 'class': 'управление проектами', 'level': 'high', 'location': 'внешнее'}"),
    ]

    # request 2
    request(llm, messages)
