from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_gigachat.chat_models import GigaChat
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")


def request(agent, config, text):
    messages = {
        "messages": [("user", text)]
    }
    response = agent.invoke(messages, config=config)
    content = response["messages"][-1].content
    print("Assistant: ", content)


if __name__ == "__main__":

    # load environment variables
    load_dotenv()

    # initialize LLM object
    llm = GigaChat(
        credentials=os.environ["auth_key"],
        scope="GIGACHAT_API_PERS",
        model="GigaChat-Pro",
        verify_ssl_certs=False,
    )

    context = Path("config/terms.txt").read_text()

    # setup system prompt
    system_prompt = ("Ты ИТ специалист, тебя зовут Архимед."
                     f"Для описания системы используются следующие параметры: {
                         context}"
                     "Твоя задача - узнать значения параметров, используемых для описания системы. Запроси эти параметры у пользователя в режиме интервью, задавая вопросы про каждый из параметров.")

    # create reactive agent (chat bot)
    agent = create_react_agent(
        llm, tools=[], checkpointer=MemorySaver(), state_modifier=system_prompt)

    # chat configuration
    config = {
        "configurable": {
            "thread_id": "123"
        }
    }

    # main chat loop
    request(agent, config, "")
    while (True):
        text = input("\nHuman: ")
        print("User: ", text)
        if text == "":
            break
        request(agent, config, text)
