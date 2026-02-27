# agent/tools.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

Route = Literal["blocked", "out_of_scope", "ambiguous", "in_scope"]

DB_DIR = "db/chroma"
COLLECTION_NAME = "junggu_guides"


# ---------- 공용 LLM ----------
def _router_llm() -> ChatOpenAI:
    # 분기/판단은 최대한 deterministic하게
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------- 1) Guard: 부적절 질문 판단 (LLM) ----------
@dataclass
class GuardResult:
    allowed: bool
    reason: str


def guard_user_input(question: str) -> GuardResult:
    """
    사전 금지어가 아니라, LLM이 '부적절/모욕/혐오/성적/폭력/개인정보 요구' 등을 보고 판단.
    - allowed=False이면 reason에 짧은 안내문을 넣어 반환.
    """
    llm = _router_llm()

    system = """\
너는 입력 안전성 필터다.
사용자 질문이 부적절한지 판단해 JSON으로만 답해라.

부적절 예시:
- 욕설/모욕/비하/혐오 표현
- 성적/폭력적/자해 조장
- 개인정보(주민번호/계좌/정확한 주소 등) 요구
- 불법 행위 조장

출력 형식(JSON only):
{"allowed": true/false, "reason": "allowed가 false인 경우에만 사용자에게 보여줄 짧은 안내문"}"""

    user = f"사용자 질문: {question}"

    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    try:
        data = json.loads(resp.content)
        allowed = bool(data.get("allowed", True))
        reason = str(data.get("reason", "")).strip()
        if not allowed and not reason:
            reason = "부적절한 표현이 포함되어 답변할 수 없어요. 표현을 바꿔서 질문해 주세요."
        return GuardResult(allowed=allowed, reason=reason)
    except Exception:
        # 파싱 실패 시 보수적으로 통과(서비스 중단 방지) + 필요하면 로그로 남기기
        return GuardResult(allowed=True, reason="")


# ---------- 2) Route: 중구 가이드 범위 판단 (LLM) ----------
@dataclass
class RouteResult:
    route: Route
    reason: str
    clarify_question: Optional[str] = None


def route_question(question: str) -> RouteResult:
    """
    사용자의 질문이:
    - out_of_scope: 서울 중구 가이드북 범위 밖(예: 부산/여수/해운대 등, 또는 중구 무관)
    - ambiguous: 지역/대상이 모호해서 되묻기가 필요한 상태
    - in_scope: 중구 가이드북 범위로 보이는 질문
    을 LLM이 판단.
    """
    llm = _router_llm()

    system = """\
너는 '서울 중구 여행 가이드북 8종' 기반 챗봇의 라우터다.
질문이 이 챗봇의 범위에 들어오는지 판단해라.

범위(in_scope):
- 서울 중구(명동/을지로/필동/장충동/광희동/회현동/소공동/중림동) 여행/관광/먹거리/볼거리/분위기/역사/코스/상점/문화 등

범위 밖(out_of_scope):
- 서울 중구가 아닌 지역(부산/해운대/여수/제주 등)
- 여행/관광과 무관한 일반 지식(예: 코딩 숙제, 수학 문제 등)
- '중구청 가는 길'처럼 실시간 길찾기/교통 경로 안내처럼
  문서 기반으로 답하기 어려운 요청도 out_of_scope로 분류 가능(단, 애매하면 ambiguous)

모호(ambiguous):
- "중구 맛집 추천"처럼 중구는 맞는데 구체성이 너무 없어서
  목적/동선/취향/예산/시간대 등을 물어봐야 답 품질이 나오는 경우
- "여기 어디야?" 같이 맥락이 부족한 경우

출력은 JSON only:
{"route":"in_scope|out_of_scope|ambiguous", "reason":"한 줄 설명", "clarify_question":"ambiguous일 때만 사용자에게 되묻는 질문(없으면 null)"}"""

    user = f"사용자 질문: {question}"

    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    try:
        data = json.loads(resp.content)
        route = data.get("route", "ambiguous")
        if route not in ("in_scope", "out_of_scope", "ambiguous"):
            route = "ambiguous"
        reason = str(data.get("reason", "")).strip() or "질문 범위를 판단하기 어려워요."
        cq = data.get("clarify_question", None)
        if cq is not None:
            cq = str(cq).strip()
            if cq.lower() in ("null", ""):
                cq = None
        return RouteResult(route=route, reason=reason, clarify_question=cq)
    except Exception:
        return RouteResult(route="ambiguous", reason="질문 범위를 판단하기 어려워요.", clarify_question="서울 중구(명동/을지로/필동/장충동/광희동/회현동/소공동/중림동) 중 어디를 기준으로 추천해드릴까요?")


# ---------- 3) Retriever / Sources ----------
def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def retrieve_docs(question: str, k: int = 5) -> List:
    vs = get_vectorstore()
    # retriever 형태로도 가능하지만, 여기서는 단순히 docs만 반환
    return vs.as_retriever(search_kwargs={"k": k}).invoke(question)


def format_sources(docs: List) -> List[str]:
    """
    docs의 metadata(source_file, page)를 사람이 읽기 좋은 출처 문자열로 변환
    """
    sources = []
    for d in docs or []:
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", None)
        if page is not None:
            sources.append(f"{src}, page {int(page) + 1}")
        else:
            sources.append(src)
    # 중복 제거(순서 유지)
    uniq = []
    seen = set()
    for s in sources:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


# ---------- 4) 답변 가능 여부 판단 (LLM) ----------
@dataclass
class AnswerabilityResult:
    answerable: bool
    reason: str
    next_route: Literal["generate", "clarify", "out_of_scope"]
    clarify_question: Optional[str] = None


def judge_answerability(question: str, docs: List) -> AnswerabilityResult:
    """
    '이 문서 컨텍스트로 지금 질문에 답할 수 있는가?'를 LLM이 판단.
    - answerable=True면 generate
    - answerable=False면 clarify 또는 out_of_scope로 보냄
    """
    llm = _router_llm()

    # 문서 컨텍스트를 너무 길게 주지 않기 (요약된 발췌만)
    snippets = []
    for i, d in enumerate(docs or [], start=1):
        text = (d.page_content or "").strip().replace("\n", " ")
        text = text[:600]  # 각 chunk 발췌 길이 제한
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", None)
        if page is not None:
            meta = f"{src} p{int(page)+1}"
        else:
            meta = src
        snippets.append(f"[{i}] ({meta}) {text}")

    context = "\n".join(snippets) if snippets else "(검색된 문서 발췌 없음)"

    system = """\
너는 '문서 기반 답변 가능 여부' 심사관이다.
사용자 질문과 제공된 문서 발췌를 보고,
- 이 발췌만으로 근거 있는 답변이 가능한지 판단하라.
- 불가능하면 (1) 되묻기(clarify)가 필요한지, (2) 문서 범위 밖(out_of_scope)인지 결정하라.

판단 기준:
- 질문이 길찾기/교통 경로/실시간 정보 등 문서로 보장 못 하는 요청이면 out_of_scope 권장
- 문서에 관련 단서가 거의 없으면 out_of_scope 또는 clarify
- 사용자의 조건(시간대/취향/예산/동행/목적)이 부족해 추천 품질이 떨어지면 clarify

출력(JSON only):
{
  "answerable": true/false,
  "next_route": "generate|clarify|out_of_scope",
  "reason": "한 줄 설명",
  "clarify_question": "clarify일 때만 사용자에게 물어볼 질문(없으면 null)"
}"""

    user = f"""\
사용자 질문: {question}

문서 발췌:
{context}
"""

    resp = llm.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    try:
        data = json.loads(resp.content)
        answerable = bool(data.get("answerable", False))
        next_route = data.get("next_route", "clarify")
        if next_route not in ("generate", "clarify", "out_of_scope"):
            next_route = "clarify"
        reason = str(data.get("reason", "")).strip() or "문서 근거가 충분한지 판단하기 어려워요."
        cq = data.get("clarify_question", None)
        if cq is not None:
            cq = str(cq).strip()
            if cq.lower() in ("null", ""):
                cq = None
        return AnswerabilityResult(
            answerable=answerable,
            next_route=next_route,
            reason=reason,
            clarify_question=cq,
        )
    except Exception:
        return AnswerabilityResult(
            answerable=False,
            next_route="clarify",
            reason="문서 근거가 충분한지 판단하기 어려워요.",
            clarify_question="어느 동(명동/을지로/필동/장충동/광희동/회현동/소공동/중림동)을 기준으로 알려드릴까요?",
        )