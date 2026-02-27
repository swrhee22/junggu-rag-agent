# agent/graph.py
from __future__ import annotations

from typing import TypedDict, Literal, List

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools import (
    guard_user_input,
    route_question,
    retrieve_docs,
    judge_answerability,
    format_sources,
)
from agent.prompts import SYSTEM_PROMPT, OUT_OF_SCOPE_PROMPT, CLARIFY_PROMPT, ANSWER_TEMPLATE_HINT

Route = Literal["blocked", "out_of_scope", "ambiguous", "in_scope"]


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    docs: list
    answer: str
    sources: List[str]
    # 추가(디버깅/보고용)
    router_reason: str
    judge_reason: str


# 1) Guard: 부적절 질문 차단 (LLM)
def guard_node(state: AgentState) -> AgentState:
    q = state["question"]
    result = guard_user_input(q)
    if not result.allowed:
        return {"route": "blocked", "answer": result.reason, "sources": [], "router_reason": "blocked_by_guard"}
    return state


# 2) Route: 중구/비중구/애매 분기 (LLM)
def route_node(state: AgentState) -> AgentState:
    q = state["question"]
    rr = route_question(q)
    if rr.route == "out_of_scope":
        return {"route": "out_of_scope", "router_reason": rr.reason}
    if rr.route == "ambiguous":
        # clarify_node에서 사용하게 메시지에 이유를 남겨둘 수도 있음
        return {"route": "ambiguous", "router_reason": rr.reason, "answer": rr.clarify_question or ""}
    return {"route": "in_scope", "router_reason": rr.reason}


# 3) Retrieve: 벡터DB 검색
def retrieve_node(state: AgentState) -> AgentState:
    q = state["question"]
    docs = retrieve_docs(q, k=5)
    return {"docs": docs, "sources": format_sources(docs)}


# 3.5) Judge: 문서 근거로 답변 가능한지 판단 (LLM)
def judge_node(state: AgentState) -> AgentState:
    q = state["question"]
    docs = state.get("docs", [])
    jr = judge_answerability(q, docs)

    if jr.next_route == "generate":
        return {"route": "in_scope", "judge_reason": jr.reason}

    if jr.next_route == "out_of_scope":
        return {"route": "out_of_scope", "judge_reason": jr.reason}

    # clarify
    # clarify_node에서 사용할 "answer"에 미리 되묻기 문장을 넣어둘 수도 있고,
    # clarify_node에서 다시 LLM으로 만들게 할 수도 있음(현재는 answer로 전달)
    return {
        "route": "ambiguous",
        "judge_reason": jr.reason,
        "answer": jr.clarify_question or "",
        "sources": [],
    }


# 4) Generate: 문서 근거 기반 답변
def generate_node(state: AgentState) -> AgentState:
    q = state["question"]
    docs = state.get("docs", [])
    sources = state.get("sources", [])

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
    # route_node 또는 judge_node에서 미리 answer(되묻기 문장)를 넣어준 경우가 있으면 우선 사용
    preset = (state.get("answer") or "").strip()
    if preset:
        return {"answer": preset, "sources": []}

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
    g.add_node("judge", judge_node)
    g.add_node("generate", generate_node)
    g.add_node("out_of_scope", out_of_scope_node)
    g.add_node("clarify", clarify_node)

    g.set_entry_point("guard")

    def after_guard(state: AgentState):
        if state.get("route") == "blocked":
            return END
        return "route"

    g.add_conditional_edges("guard", after_guard)

    def after_route(state: AgentState):
        r = state.get("route")
        if r == "out_of_scope":
            return "out_of_scope"
        if r == "ambiguous":
            return "clarify"
        return "retrieve"

    g.add_conditional_edges("route", after_route)

    g.add_edge("retrieve", "judge")

    def after_judge(state: AgentState):
        r = state.get("route")
        if r == "out_of_scope":
            return "out_of_scope"
        if r == "ambiguous":
            return "clarify"
        return "generate"

    g.add_conditional_edges("judge", after_judge)

    g.add_edge("generate", END)
    g.add_edge("out_of_scope", END)
    g.add_edge("clarify", END)

    return g.compile()


_graph = build_graph()


def run_agent(question: str) -> tuple[str, List[str], Route]:
    state = _graph.invoke({"question": question})
    return state.get("answer", ""), state.get("sources", []), state.get("route", "in_scope")