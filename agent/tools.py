# agent/tools.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Dict

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


Route = Literal["blocked", "out_of_scope", "ambiguous", "in_scope"]

DB_DIR = "db/chroma"
COLLECTION_NAME = "junggu_guides"


# =========================
# 1) Retrieval helpers
# =========================

DONG_TO_PDF = {
    "명동": "명성답게 빛나는 동네 명동.pdf",
    "을지로": "옛것 을지 금으로 을지로.pdf",
    "필동": "필름처럼 새겨지는 감성동네 필동.pdf",
    "장충동": "우리가 몰랐던 리얼장충 장충동.pdf",
    "광희동": "세계가 열광하는 환희의빛 광희동.pdf",
    "회현동": "기회를 현재로 일구어낸 회현동.pdf",
    "소공동": "서울소울 피어나는 성공로드 소공동.pdf",
    "중림동": "소중히 마음에 담는그림 중림동.pdf",
}

def infer_dong_filter(question: str) -> Optional[Dict]:
    """
    질문에 특정 동이 명시되면 해당 PDF로만 검색하도록 metadata filter 반환.
    없으면 None(= 전체 검색).
    """
    for dong, pdf in DONG_TO_PDF.items():
        if dong in question:
            return {"source_file": pdf}
    return None

def infer_dong_name(question: str) -> str | None:
    for dong in DONG_TO_PDF.keys():
        if dong in question:
            return dong
    return None

def build_safe_alternatives(question: str) -> list[str]:
    """
    '문서로 답 가능한' 대안 질문만 고정 템플릿으로 생성.
    길찾기/대중교통/운영시간/가격/전화번호 같은 실시간/정확정보는 절대 제안하지 않는다.
    """
    dong = infer_dong_filter(question)
    scope = dong if dong else "중구(명동/을지로/필동/장충동/광희동/회현동/소공동/중림동)"

    return [
        f"{scope}에서 가볼만한 곳 추천해줘",
        f"{scope} 맛집 추천해줘",
        f"{scope} 카페 추천해줘",
        f"{scope} 반나절 코스 짜줘",
        f"{scope} 분위기/특징 알려줘",
    ]


def get_retriever(k: int = 5, metadata_filter: dict | None = None):
    """
    Chroma 로컬 DB에서 retriever를 반환.
    - 문서 임베딩은 이미 db에 저장되어 있고,
    - 여기 embeddings는 '질문(query) 임베딩'을 위해 필요함.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    search_kwargs = {"k": k}
    if metadata_filter:
        # 대부분 환경에서 filter로 동작
        search_kwargs["filter"] = metadata_filter
        # 만약 네 환경에서 filter가 안 먹으면 아래로 교체:
        # search_kwargs["where"] = metadata_filter

    return vectordb.as_retriever(search_kwargs=search_kwargs)


def format_sources(docs) -> List[str]:
    """
    Streamlit expander에 보여줄 출처 문자열 리스트.
    build_index.py에서 metadata["source_file"]을 넣어둔 전제.
    """
    out = []
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        fname = meta.get("source_file") or meta.get("source") or "unknown"
        page = meta.get("page")
        if page is not None:
            out.append(f"{fname}, page {int(page)+1}")
        else:
            out.append(f"{fname}")

    # 중복 제거(순서 유지)
    uniq, seen = [], set()
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# =========================
# 2) Judge (LLM) with Pydantic validation
# =========================

class JudgeOut(BaseModel):
    route: Route
    allowed: bool
    needs_soft_tone: bool
    answerable: bool
    message: str
    clarify_question: Optional[str] = None


@dataclass
class JudgeResult:
    route: Route
    allowed: bool
    needs_soft_tone: bool
    answerable: bool
    message: str
    clarify_question: Optional[str] = None


_JUDGE_SYSTEM = """\
너는 '서울 중구 가이드북 8종' 기반 Q&A 챗봇의 입력 심사(가드) + 라우팅 담당자다.

[최우선 규칙 - blocked]
사용자 입력에 아래가 포함되면 무조건 blocked로 판단한다.
- 한국어 욕설/비하/혐오/모욕/성적 비하/가학적 표현
- 욕설의 축약/초성/은어/변형 포함 (예: ㅅㅂ, ㅂㅅ, ㅈㄴ, ㅈㄹ, ㅆㅂ, 시1발, 씨8, 병1신 등)
- 상대를 직접 모욕하는 말투(“너 멍청하냐”, “닥쳐”, “꺼져”)도 blocked로 본다.
- 자해/타해 의도도 blocked

[범위 규칙 - out_of_scope / ambiguous / in_scope]
- 이 시스템은 '서울 중구(명동/을지로/필동/장충동/광희동/회현동/소공동/중림동)' 관련 질문만 in_scope로 본다.
- 서울 중구 외 지역(부산/제주/여수/강남 등) 관련이면 out_of_scope.
- 질문이 너무 짧거나 의도가 불명확하면 ambiguous.
  단, “명동 가볼만한 곳 추천”처럼 동+의도가 있으면 ambiguous로 보내지 말고 in_scope로 둔다.
- in_scope라도 '문서로 답하기 어려운 유형'(길찾기/실시간/정확한 운영시간/최신 가격/전화번호 등)이면 answerable=false로 둔다.

[출력 형식 - 엄격]
반드시 JSON만 출력한다. 설명/코드블록/추가 텍스트 금지.
각 필드는 반드시 아래 타입을 지켜라:
- route: 문자열(정해진 4개 중 1개)
- allowed: boolean (true/false)  ← 문자열 "false" 금지
- needs_soft_tone: boolean
- answerable: boolean
- message: 문자열
- clarify_question: 문자열 또는 null

작성 규칙:
- blocked: allowed=false, answerable=false, message는 정중한 차단 안내 1문장, clarify_question=null
- out_of_scope: allowed=true, answerable=false, message는 범위 안내 1~2문장, clarify_question=null
- ambiguous: allowed=true, answerable=false, message 1문장, clarify_question은 되묻기 1문장(선택지 포함 권장)
- in_scope: allowed=true, answerable=true, message는 빈 문자열("")로, clarify_question=null

예시(판단 기준 참고):
입력: "씨발" → blocked
입력: "ㅅㅂ" → blocked
입력: "부산 해운대 맛집" → out_of_scope
입력: "명동 가볼만한 곳 추천" → in_scope
입력: "맛집 추천" → ambiguous
"""


def judge_question(question: str) -> JudgeResult:
    """
    LLM이 JSON(진짜 boolean 포함)을 출력하도록 '구조화 출력'로 강제하고,
    Pydantic으로 타입 검증까지 수행한다.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    llm_structured = llm.with_structured_output(JudgeOut)

    try:
        out: JudgeOut = llm_structured.invoke([
            ("system", _JUDGE_SYSTEM),
            ("user", f"사용자 입력: {question}")
        ])

    except ValidationError:
        # 스키마를 어기거나 타입이 이상하면 안전한 fallback
        return JudgeResult(
            route="ambiguous",
            allowed=True,
            needs_soft_tone=False,
            answerable=False,
            message="질문을 조금만 더 구체적으로 알려주세요.",
            clarify_question="어느 동(명동/을지로/필동/장충동/광희동/회현동/소공동/중림동) 기준으로, 무엇을 추천받고 싶으신가요?"
        )

    # 최소 보정(빈 값 대비)
    if out.route == "ambiguous" and not out.clarify_question:
        out.clarify_question = "어느 동(명동/을지로/필동/장충동/광희동/회현동/소공동/중림동) 기준으로, 무엇을 추천받고 싶으신가요?"

    if out.route in ("blocked", "out_of_scope", "ambiguous") and not out.message:
        # route별 기본 message
        if out.route == "blocked":
            out.message = "부적절한 표현이 포함되어 답변할 수 없어요. 표현을 바꿔 다시 질문해 주세요."
        elif out.route == "out_of_scope":
            out.message = "현재는 ‘서울 중구 8개 동 가이드북’ 범위에서만 답변할 수 있어요. 중구 기준으로 질문을 바꿔 주세요."
        else:
            out.message = "질문을 조금만 더 구체적으로 알려주세요."

    return JudgeResult(
        route=out.route,
        allowed=out.allowed,
        needs_soft_tone=out.needs_soft_tone,
        answerable=out.answerable,
        message=out.message,
        clarify_question=out.clarify_question
    )