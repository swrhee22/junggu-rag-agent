from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, List

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage, HumanMessage


DB_DIR = "db/chroma"
COLLECTION_NAME = "junggu_guides"


# -----------------------------
# Guard (Hard block / Soft tone)
# -----------------------------

GuardAction = Literal["allow", "soft", "block"]


@dataclass
class GuardResult:
    action: GuardAction
    message: str


# 하드블락: 강한 욕설/혐오/성적 비하 등 “운영상 차단” 수준만
HARD_BLOCK_PATTERNS = [
    r"\b씨발\b", r"\b시발\b", r"\b병신\b", r"\b좆\b", r"\b좆같\b",
    r"\b개새끼\b", r"\b썅\b",
]

# 소프트: 무례/짜증/비난투(차단까지는 X, 톤만 정리)
SOFT_TONE_PATTERNS = [
    r"\b니가\b", r"\b너가\b", r"\b뭐하냐\b", r"\b멍청\b", r"\b븅\b",
]


def _match_any(patterns: List[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def guard_user_input(question: str) -> GuardResult:
    q = question.strip()

    # 1) 하드블락 룰 우선
    if _match_any(HARD_BLOCK_PATTERNS, q):
        return GuardResult(
            action="block",
            message="부적절한 표현이 포함되어 있어 답변할 수 없어요. 표현을 바꿔서 질문해 주세요."
        )

    # 2) 소프트 톤 룰 (차단 X)
    if _match_any(SOFT_TONE_PATTERNS, q):
        return GuardResult(
            action="soft",
            message="사용자 표현이 다소 거칠 수 있음 → 정중하고 차분하게 응대"
        )

    # 3) 그 외는 모델로 한 번 더 판정(원하는 ‘모델 판단’ 반영)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    sys = SystemMessage(content=(
        "너는 사용자 입력의 적절성을 분류하는 필터다.\n"
        "분류는 오직 다음 중 하나로만 출력하라: ALLOW / SOFT / BLOCK\n"
        "- BLOCK: 강한 욕설, 혐오/차별, 성적 비하, 폭력/자해 선동 등 운영상 차단이 필요한 경우\n"
        "- SOFT: 무례/비난/짜증, 공격적 말투(그러나 차단까지는 아닌 경우)\n"
        "- ALLOW: 그 외 정상 질문\n"
        "추가 설명 없이 한 단어로만 출력하라."
    ))
    resp = llm.invoke([sys, HumanMessage(content=q)])
    label = (resp.content or "").strip().upper()

    if label == "BLOCK":
        return GuardResult("block", "부적절한 표현이 포함되어 있어 답변할 수 없어요. 표현을 바꿔서 질문해 주세요.")
    if label == "SOFT":
        return GuardResult("soft", "사용자 표현이 다소 거칠 수 있음 → 정중하고 차분하게 응대")
    return GuardResult("allow", "")


# -----------------------------
# Retriever
# -----------------------------

def get_retriever(k: int = 5):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    return vectordb.as_retriever(search_kwargs={"k": k})


def format_sources(docs) -> List[str]:
    sources: List[str] = []
    for d in docs:
        meta = d.metadata or {}
        f = meta.get("source_file", "unknown.pdf")
        page = meta.get("page", None)
        if page is not None:
            sources.append(f"{f} (page {page+1})")
        else:
            sources.append(f"{f}")
    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for s in sources:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


# -----------------------------
# LLM-based routing (clarify 좁히기)
# -----------------------------

Route = Literal["out_of_scope", "ambiguous", "in_scope"]


def route_question_llm(question: str) -> Route:
    """
    - in_scope: 중구 8개 동/명동/을지로/필동/장충/광희/회현/소공/중림 관련 질문.
      목적이 불명확해도(‘명동 가볼만한 곳’) 기본은 in_scope로 두고 답하게 함.
    - ambiguous: 지역/범위/의도 자체가 너무 불명확해서 되물어야 하는 경우
      (예: '추천해줘', '맛집 알려줘'인데 지역 언급 없음)
    - out_of_scope: 명백히 중구가 아닌 지역/주제(해운대, 여수, 제주 등)
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    sys = SystemMessage(content=(
        "너는 질문을 3가지 중 하나로 라우팅한다: IN_SCOPE / AMBIGUOUS / OUT_OF_SCOPE.\n"
        "IN_SCOPE 기준:\n"
        "- 서울 '중구' 또는 중구의 대표 지역(명동, 을지로, 필동, 장충동, 광희동, 회현동, 소공동, 중림동)이 언급되면 기본은 IN_SCOPE.\n"
        "- 목적이 불명확해도(예: '명동 가볼만한 곳') IN_SCOPE로 두고 일단 답변하도록 한다.\n"
        "OUT_OF_SCOPE 기준:\n"
        "- 중구와 무관한 다른 지역(예: 부산 해운대, 여수, 제주 등) 또는 명백히 다른 주제.\n"
        "AMBIGUOUS 기준:\n"
        "- 지역/범위가 전혀 없고, 무엇을 원하는지 너무 모호해서 질문을 좁혀야 할 때.\n"
        "출력은 오직 한 단어로만: IN_SCOPE / AMBIGUOUS / OUT_OF_SCOPE"
    ))

    resp = llm.invoke([sys, HumanMessage(content=question)])
    label = (resp.content or "").strip().upper()

    if label == "OUT_OF_SCOPE":
        return "out_of_scope"
    if label == "AMBIGUOUS":
        return "ambiguous"
    return "in_scope"


# -----------------------------
# Answerability judge (문서로 답 가능?)
# -----------------------------

def judge_answerability_llm(question: str, docs) -> bool:
    """
    '중구 관련'이라도,
    - 길찾기/교통/실시간 정보/구청 가는 법 등
    - 가이드북 PDF에 없을 확률이 높은 질문
    을 문서 근거로 답할 수 있는지 판정.
    """
    # 검색 결과가 거의 없으면 답변 어렵다고 판단
    if not docs:
        return False

    context = "\n\n".join([d.page_content for d in docs[:5]])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    sys = SystemMessage(content=(
        "너는 '주어진 문서 발췌로 질문에 답할 수 있는지'만 판정한다.\n"
        "규칙:\n"
        "- 문서 발췌에 직접 근거가 있으면 YES\n"
        "- 근거가 부족하거나, 문서 밖 지식/실시간 정보/길찾기/교통/운영시간 단정 등이 필요하면 NO\n"
        "출력은 오직 YES 또는 NO"
    ))
    user = HumanMessage(content=f"질문: {question}\n\n문서 발췌:\n{context}")
    resp = llm.invoke([sys, user])
    return (resp.content or "").strip().upper() == "YES"