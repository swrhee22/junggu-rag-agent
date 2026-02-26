# agent/graph.py
from __future__ import annotations

from typing import TypedDict, Literal, List

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools import get_retriever, guard_user_input, is_about_junggu, is_out_of_scope, format_sources
from agent.prompts import SYSTEM_PROMPT, OUT_OF_SCOPE_PROMPT, CLARIFY_PROMPT, ANSWER_TEMPLATE_HINT


Route = Literal["blocked", "out_of_scope", "ambiguous", "in_scope"]


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    docs: list
    answer: str
    sources: List[str]


# 1) Guard: 부적절 질문 차단
def guard_node(state: AgentState) -> AgentState:
    q = state["question"]
    result = guard_user_input(q)
    if not result.allowed:
        return {"route": "blocked", "answer": result.reason, "sources": []}
    return state


# 2) Route: 중구/비중구/애매 분기
def route_node(state: AgentState) -> AgentState:
    q = state["question"]

    # 명시적으로 다른 지역이면 out_of_scope
    if is_out_of_scope(q):
        return {"route": "out_of_scope"}

    # 중구/8개 동이 명시되면 in_scope
    if is_about_junggu(q):
        return {"route": "in_scope"}

    # 애매하면 clarifying 질문으로
    return {"route": "ambiguous"}


# 3) Retrieve: 벡터DB 검색
def retrieve_node(state: AgentState) -> AgentState:
    q = state["question"]
    retriever = get_retriever(k=5)
    docs = retriever.invoke(q)
    return {"docs": docs, "sources": format_sources(docs)}


# 4) Generate: 문서 근거 기반 답변
def generate_node(state: AgentState) -> AgentState:
    q = state["question"]
    docs = state.get("docs", [])
    sources = state.get("sources", [])

    # 컨텍스트(문서 chunk들)
    context = "\n\n".join([f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1)])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""\
질문: {q}

아래는 참고 문서 발췌(컨텍스트)입니다. 이 내용에 근거해서만 답변하세요.
컨텍스트:
{context}

{ANSWER_TEMPLATE_HINT}

주의:
- 문서에 근거 없는 내용은 단정하지 말고, 필요하면 추가 질문을 하세요.
""")
    ]
    resp = llm.invoke(messages)
    return {"answer": resp.content, "sources": sources}


# 5) Out-of-scope 안내
def out_of_scope_node(state: AgentState) -> AgentState:
    q = state["question"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    resp = llm.invoke([
        SystemMessage(content=OUT_OF_SCOPE_PROMPT),
        HumanMessage(content=f"사용자 질문: {q}")
    ])
    return {"answer": resp.content, "sources": []}


# 6) Clarify(되묻기)
def clarify_node(state: AgentState) -> AgentState:
    q = state["question"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    resp = llm.invoke([
        SystemMessage(content=CLARIFY_PROMPT),
        HumanMessage(content=f"사용자 질문: {q}")
    ])
    return {"answer": resp.content, "sources": []}


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("guard", guard_node)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("out_of_scope", out_of_scope_node)
    g.add_node("clarify", clarify_node)

    g.set_entry_point("guard")

    # guard 이후: blocked면 종료, 아니면 route로
    def after_guard(state: AgentState):
        if state.get("route") == "blocked":
            return END
        return "route"

    g.add_conditional_edges("guard", after_guard)

    # route 이후: out_of_scope / ambiguous / in_scope 분기
    def after_route(state: AgentState):
        r = state.get("route")
        if r == "out_of_scope":
            return "out_of_scope"
        if r == "ambiguous":
            return "clarify"
        return "retrieve"

    g.add_conditional_edges("route", after_route)

    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    g.add_edge("out_of_scope", END)
    g.add_edge("clarify", END)

    return g.compile()


_graph = build_graph()


def run_agent(question: str) -> tuple[str, List[str], Route]:
    """
    app.py에서 쓰기 편하게:
    - answer
    - sources(출처 리스트)
    - route(어떤 경로로 처리됐는지)
    """
    state = _graph.invoke({"question": question})
    return state.get("answer", ""), state.get("sources", []), state.get("route", "in_scope")