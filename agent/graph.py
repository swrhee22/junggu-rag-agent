from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools import get_retriever
from agent.prompts import SYSTEM_PROMPT


def run_agent(question: str):
    retriever = get_retriever()
    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""
[질문]
{question}

[문서]
{context}

위 문서를 기반으로 답변하세요.
""")
    ]

    response = llm.invoke(messages)

    return response.content, docs