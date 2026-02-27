from __future__ import annotations

from typing import TypedDict, Literal, List

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools import (
    get_retriever,
    guard_user_input,
    route_question_llm,
    judge_answerability_llm,
    format_sources,
)
from agent.prompts import (
    SYSTEM_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    CLARIFY_PROMPT,
    ANSWER_TEMPLATE_HINT,
    CANNOT_ANSWER_PROMPT,
)

Route = Literal["blocked", "out_of_scope", "ambiguous", "in_scope", "cannot_answer"]


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    docs: list
    answer: str
    sources: List[str]
    soft_note: str  # 소프트 가드(무례/짜증)일 때 톤 완화 안내문


# 1) Guard: 하드블락 / 소프트 처리 분기
def guard_node(state: AgentState) -> AgentState:
    q = state["question"]
    result = guard_user_input(q)

    if result.action == "block":
        return {"route": "blocked", "answer": result.message, "sources": []}

    if result.action == "soft":
        # 차단하지 않고, 이후 답변에서 톤만 정리하도록 note 저장
        return {"soft_note": result.message}

    return state


# 2) Route: LLM 기반 라우팅(clarify 조건 좁힘)
def route_node(state: AgentState) -> AgentState:
    q = state["question"]
    r = route_question_llm(q)
    return {"route": r}


# 3) Retrieve: 벡터DB 검색
def retrieve_node(state: AgentState) -> AgentState:
    q = state["question"]
    retriever = get_retriever(k=5)
    docs = retriever.invoke(q)
    return {"docs": docs, "sources": format_sources(docs)}


# 4) Answerability Judge: 이 질문을 "현재 문서로" 답할 수 있는지 판정
def answerability_node(state: AgentState) -> AgentState:
    q = state["question"]
    docs = state.get("docs", [])
    ok = judge_answerability_llm(q, docs)
    if not ok:
        return {"route": "cannot_answer"}
    return state


# 5) Generate: 문서 근거 기반 답변
def generate_node(state: AgentState) -> AgentState:
    q = state["question"]
    docs = state.get("docs", [])
    sources = state.get("sources", [])
    soft_note = state.get("soft_note")

    context = "\n\n".join([f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1)])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # 소프트 가드가 있으면, 답변 서두에 톤 완화 문구를 붙이도록 힌트
    tone_hint = ""
    if soft_note:
        tone_hint = f"\n\n추가 지시: 사용자의 표현이 다소 거칠 수 있으니, 정중하고 차분하게 응대하세요. (내부 메모: {soft_note})\n"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT + tone_hint),
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


# 6) Out-of-scope 안내
def out_of_scope_node(state: AgentState) -> AgentState:
    q = state["question"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    resp = llm.invoke([
        SystemMessage(content=OUT_OF_SCOPE_PROMPT),
        HumanMessage(content=f"사용자 질문: {q}")
    ])
    return {"answer": resp.content, "sources": []}


# 7) Clarify(되묻기)
def clarify_node(state: AgentState) -> AgentState:
    q = state["question"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    resp = llm.invoke([
        SystemMessage(content=CLARIFY_PROMPT),
        HumanMessage(content=f"사용자 질문: {q}")
    ])
    return {"answer": resp.content, "sources": []}


# 8) Cannot Answer(문서 근거 부족)
def cannot_answer_node(state: AgentState) -> AgentState:
    q = state["question"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    resp = llm.invoke([
        SystemMessage(content=CANNOT_ANSWER_PROMPT),
        HumanMessage(content=f"사용자 질문: {q}")
    ])
    return {"answer": resp.content, "sources": []}


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("guard", guard_node)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answerability", answerability_node)
    g.add_node("generate", generate_node)
    g.add_node("out_of_scope", out_of_scope_node)
    g.add_node("clarify", clarify_node)
    g.add_node("cannot_answer", cannot_answer_node)

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
        # in_scope면 retrieve로
        return "retrieve"

    g.add_conditional_edges("route", after_route)

    # retrieve 후 answerability 체크 → ok면 generate / 아니면 cannot_answer
    def after_retrieve(state: AgentState):
        return "answerability"

    g.add_edge("retrieve", "answerability")

    def after_answerability(state: AgentState):
        if state.get("route") == "cannot_answer":
            return "cannot_answer"
        return "generate"

    g.add_conditional_edges("answerability", after_answerability)

    g.add_edge("generate", END)
    g.add_edge("out_of_scope", END)
    g.add_edge("clarify", END)
    g.add_edge("cannot_answer", END)

    return g.compile()


_graph = build_graph()


def run_agent(question: str) -> tuple[str, List[str], Route]:
    state = _graph.invoke({"question": question})
    return (
        state.get("answer", ""),
        state.get("sources", []),
        state.get("route", "in_scope"),
    )