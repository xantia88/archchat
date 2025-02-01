from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from langchain_core.output_parsers import StrOutputParser
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

    # create prompt
    text = ("Расположение системы описывается параметром 'location', расположение может быть либо внутренним либо внешним."
            "Система описывается как объект с типом 'system'."
            "Уровень критичности системы описывается параметром 'level'."
            "Класс системы описывается параметром 'class'."
            "Класс системы может принимать одно из следующих значений: управление проектами, бухгалтерия.")

    context = f"Используй следующий контекст: {text}"
    command = "Cоставь краткое текстовое описание системы"
    data = {
        "type": "system",
        "name": "моя система",
        "class": "управление проектами",
        "level": "high",
        "location": "внешнее"
    }

    messages = [
        SystemMessage(f"{context}. {command}"),
        HumanMessage(str(data))
    ]

    # execute request
    response = llm.invoke(messages)
    parser = StrOutputParser()
    text = parser.invoke(response)
    print(text)
