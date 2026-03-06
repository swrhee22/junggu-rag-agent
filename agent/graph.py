# agent/graph.py
from __future__ import annotations

from typing import TypedDict, Literal, List

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools import (
    get_retriever,
    infer_dong_filter,
    judge_question,
    format_sources,
    build_safe_alternatives,
    infer_dong_name
)

from agent.prompts import SYSTEM_PROMPT, OUT_OF_SCOPE_PROMPT, CLARIFY_PROMPT, ANSWER_TEMPLATE_HINT


Route = Literal["blocked", "out_of_scope", "ambiguous", "in_scope", "unanswerable"]


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    docs: list
    answer: str
    sources: List[str]
    needs_soft_tone: bool
    answerable: bool
    clarify_question: str


# 1) Judge: LLM + Pydantic 검증 기반 guard/route/answerable 판정
def judge_node(state: AgentState) -> AgentState:
    q = state["question"]
    jr = judge_question(q)

    # blocked/out_of_scope/ambiguous면 여기서 answer 확정하고 종료 경로로
    if jr.route in ("blocked", "out_of_scope", "ambiguous"):
        return {
            "route": jr.route,
            "answer": (jr.clarify_question or jr.message) if jr.route == "ambiguous" else jr.message,
            "sources": [],
            "needs_soft_tone": jr.needs_soft_tone,
            "answerable": jr.answerable,
            "clarify_question": jr.clarify_question or "",
        }

    return {
        "route": "in_scope",
        "needs_soft_tone": jr.needs_soft_tone,
        "answerable": jr.answerable,
        "clarify_question": jr.clarify_question or "",
    }


# 2) Retrieve: 동이 명시되면 해당 PDF로만 검색(메타데이터 필터)
def retrieve_node(state: AgentState) -> AgentState:
    q = state["question"]
    metadata_filter = infer_dong_filter(q)
    retriever = get_retriever(k=5, metadata_filter=metadata_filter)
    docs = retriever.invoke(q)
    return {"docs": docs, "sources": format_sources(docs)}


def unanswerable_node(state: AgentState) -> AgentState:
    q = state["question"]

    dong = infer_dong_name(q)
    scope = dong if dong else "중구"

    msg = (
        "현재 시스템은 '서울 중구 가이드북 8종' 문서 내용으로만 답변합니다. "
        "그래서 길찾기/대중교통/실시간 정보(운영시간·가격·전화번호 등)처럼 "
        "문서로 단정하기 어려운 질문은 정확히 답변하기 어렵습니다.\n\n"
        "대신 아래처럼 질문을 바꿔보세요:\n"
        f"1. {scope} 가볼만한 곳 추천해줘\n"
        f"2. {scope} 맛집 추천해줘\n"
        f"3. {scope} 카페 추천해줘"
    )

    return {
        "route": "unanswerable",
        "answer": msg,
        "sources": []
    }

# 3) Generate: 문서 근거 기반 답변
def generate_node(state: AgentState) -> AgentState:
    q = state["question"]
    docs = state.get("docs", [])
    sources = state.get("sources", [])
    needs_soft_tone = bool(state.get("needs_soft_tone", False))
    answerable = bool(state.get("answerable", True))

    # 문서로 답하기 어려운 질문이면 “질문 재구성” 유도
    if not answerable:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        resp = llm.invoke([
            SystemMessage(content=OUT_OF_SCOPE_PROMPT),
            HumanMessage(content=f"""\
사용자 질문: {q}

단, 지역은 서울 중구 범위일 수 있으나 '가이드북 문서 근거'로 확정 답변이 어려운 유형이다.
가이드북 기반으로 답할 수 있도록 질문을 어떻게 바꾸면 좋을지 안내해라.
""")
        ])
        return {"answer": resp.content, "sources": []}

    context = "\n\n".join([f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1)])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    tone_hint = ""
    if needs_soft_tone:
        tone_hint = "\n추가 지시: 사용자가 다소 무례/짜증 섞인 톤이다. 정중하고 차분하게 응대하되 훈계하지 말 것.\n"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""\
질문: {q}

아래는 참고 문서 발췌(컨텍스트)입니다. 이 내용에 근거해서만 답변하세요.
컨텍스트:
{context}

{ANSWER_TEMPLATE_HINT}
{tone_hint}

주의:
- 문서에 근거 없는 내용은 단정하지 말고, 필요하면 추가 질문을 하세요.
""")
    ]
    resp = llm.invoke(messages)
    return {"answer": resp.content, "sources": sources}


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("judge", judge_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node('unanswerable', unanswerable_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("judge")

    def after_judge(state: AgentState):
        r = state.get("route")
        if r in ("blocked", "out_of_scope", "ambiguous"):
            return END

        if r == 'in_scope' and state.get('answerable') is False:
            return 'unanswerable'

        return "retrieve"

    g.add_conditional_edges("judge", after_judge)
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    g.add_edge('unanswerable', END)

    return g.compile()


_graph = build_graph()


def run_agent(question: str) -> tuple[str, List[str], Route]:
    state = _graph.invoke({"question": question})
    return state.get("answer", ""), state.get("sources", []), state.get("route", "in_scope")