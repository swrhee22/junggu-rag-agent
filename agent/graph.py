# agent/graph.py
from __future__ import annotations

from typing import TypedDict, Literal, List

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from agent.tools import get_retriever, format_sources, judge_question, infer_dong_filter
from agent.prompts import SYSTEM_PROMPT, OUT_OF_SCOPE_PROMPT, CLARIFY_PROMPT, ANSWER_TEMPLATE_HINT


Route = Literal["blocked", "out_of_scope", "ambiguous", "in_scope"]


class AgentState(TypedDict, total=False):
    question: str
    route: Route
    docs: list
    answer: str
    sources: List[str]
    # judge 결과를 state에 저장(디버깅/UX 용)
    needs_soft_tone: bool
    answerable: bool
    clarify_question: str


# 1) Judge: LLM이 guard/route/answerability 판정
def judge_node(state: AgentState) -> AgentState:
    q = state["question"]
    jr = judge_question(q)

    # blocked/out_of_scope/ambiguous면 여기서 바로 메시지 확정
    if jr.route in ("blocked", "out_of_scope", "ambiguous"):
        return {
            "route": jr.route,
            "answer": jr.message if jr.route != "ambiguous" else (jr.clarify_question or jr.message),
            "sources": [],
            "needs_soft_tone": jr.needs_soft_tone,
            "answerable": jr.answerable,
            "clarify_question": jr.clarify_question or "",
        }

    # in_scope인 경우만 다음 단계로
    return {
        "route": "in_scope",
        "needs_soft_tone": jr.needs_soft_tone,
        "answerable": jr.answerable,
        "clarify_question": jr.clarify_question or "",
    }


# 2) Retrieve
def retrieve_node(state: AgentState) -> AgentState:
    q = state["question"]
    metadata_filter = infer_dong_filter(q)  # ✅ 여기서 동 감지해서 필터 생성
    retriever = get_retriever(k=5, metadata_filter=metadata_filter)
    docs = retriever.invoke(q)
    return {"docs": docs, "sources": format_sources(docs)}


# 3) Generate
def generate_node(state: AgentState) -> AgentState:
    q = state["question"]
    docs = state.get("docs", [])
    sources = state.get("sources", [])
    needs_soft_tone = bool(state.get("needs_soft_tone", False))
    answerable = bool(state.get("answerable", True))

    # answerable=false면: “문서로 답 어렵다 + 질문 재구성 유도”로 처리
    # (그래도 route는 in_scope 유지)
    if not answerable:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        resp = llm.invoke([
            SystemMessage(content=OUT_OF_SCOPE_PROMPT),
            HumanMessage(content=f"사용자 질문: {q}\n\n단, 지역은 서울 중구 범위일 수 있으나 문서로 답하기 어려운 유형이다. 문서 기반으로 답할 수 있게 질문을 재구성하도록 안내해라.")
        ])
        return {"answer": resp.content, "sources": []}

    context = "\n\n".join([f"[{i}] {d.page_content}" for i, d in enumerate(docs, start=1)])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    tone_hint = ""
    if needs_soft_tone:
        tone_hint = "\n추가 지시: 사용자가 다소 무례/짜증 섞인 톤이므로, 정중하고 차분하게 대답하되 훈계하지 말 것.\n"

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
    g.add_node("generate", generate_node)

    g.set_entry_point("judge")

    # judge 결과에 따라: 바로 종료 or retrieve로
    def after_judge(state: AgentState):
        r = state.get("route")
        if r in ("blocked", "out_of_scope", "ambiguous"):
            return END
        return "retrieve"

    g.add_conditional_edges("judge", after_judge)

    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    return g.compile()


_graph = build_graph()


def run_agent(question: str) -> tuple[str, List[str], Route]:
    state = _graph.invoke({"question": question})
    return state.get("answer", ""), state.get("sources", []), state.get("route", "in_scope")