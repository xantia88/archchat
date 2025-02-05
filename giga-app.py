from pathlib import Path
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
    text = Path("config/R11.txt").read_text()

"""













Параметр "monitoring" содержит описание програмного обеспечения, которое используется для мониторинга системы.
"""

context = f"Используй следующий контекст: {text}"
data = "составь определение параметра monitoring"

messages = [
    SystemMessage(f"{context}"),
    HumanMessage(str(data))
]

# execute request
response = llm.invoke(messages)
parser = StrOutputParser()
text = parser.invoke(response)
print(text)
