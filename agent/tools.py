# agent/tools.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


Route = Literal["blocked", "out_of_scope", "ambiguous", "in_scope"]

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
    질문에 특정 '동'이 명시되면 해당 PDF로만 검색하도록 필터를 반환.
    없으면 None 반환(= 전체 검색).
    """
    for dong, pdf in DONG_TO_PDF.items():
        if dong in question:
            # Chroma/LangChain에서 보통 filter 키로 메타데이터 필터를 받음
            return {"source_file": pdf}
    return None

@dataclass
class JudgeResult:
    route: Route
    allowed: bool
    # “욕설/비난/무례” 등으로 인해 답변은 하되 톤을 정리할지 여부
    needs_soft_tone: bool
    # 질문이 ‘중구’이긴 한데 문서로는 답이 안 나올 확률이 높은지
    answerable: bool

    # 사용자에게 보여줄 메시지(차단/범위외/되묻기 상황)
    message: str

    # ambiguous일 때, 다음에 무엇을 물어볼지(되묻기 프롬프트)
    clarify_question: Optional[str] = None


# --- Vector DB ---
DB_DIR = "db/chroma"
COLLECTION_NAME = "junggu_guides"


def get_retriever(k: int = 5, metadata_filter: dict | None = None):
    """
    Chroma 로컬 DB에서 retriever를 반환.
    - metadata_filter가 있으면 해당 범위로만 검색
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    search_kwargs = {"k": k}
    if metadata_filter:
        # 대부분의 LangChain Chroma는 'filter' 키로 메타데이터 필터를 받음
        search_kwargs["filter"] = metadata_filter

        # 만약 네 환경에서 filter가 안 먹으면(버전 차이),
        # search_kwargs["where"] = metadata_filter 로 바꿔야 하는 경우도 있음.
        # (그때 에러 메시지 보고 정확히 맞춰줄게)

    return vectordb.as_retriever(search_kwargs=search_kwargs)


def format_sources(docs) -> List[str]:
    """
    Streamlit에 보여줄 출처 문자열 리스트로 변환.
    build_index.py에서 metadata['source_file']을 넣어뒀다는 전제.
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
    uniq = []
    seen = set()
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


# --- LLM Judge (NO regex rules) ---
_JUDGE_SYSTEM = """\
너는 '서울 중구 가이드북 8종' 기반 Q&A 챗봇의 입력 심사(가드) + 라우팅 담당자다.

# 최우선 규칙(가드)
사용자 입력에 아래가 포함되면 무조건 blocked로 판단한다.
- 한국어 욕설/비하/혐오/모욕/성적 비하/가학적 표현
- 욕설의 축약/초성/은어/변형 포함 (예: ㅅㅂ, ㅂㅅ, ㅈㄴ, ㅈㄹ, ㅆㅂ, ㅈㅅ, 시1발, 씨8, 병1신 등)
- 상대를 직접 모욕하는 말투(예: “너 멍청하냐”, “닥쳐”, “꺼져”)도 blocked
※ 사용자가 실제로 누구를 해치겠다는 의도(자해/타해)가 있으면 역시 blocked

# 범위 규칙(라우팅)
- 이 시스템은 '서울 중구(명동/을지로/필동/장충동/광희동/회현동/소공동/중림동)' 관련 질문만 in_scope로 본다.
- 서울 중구 외 지역(부산/제주/여수/인천/대구/강남 등)이나 중구 외 행정/교통/길찾기/실시간 정보(“중구청 가는 길”, “오늘 영업하나”)는 문서 기반으로 확정 답변이 어렵다면 out_of_scope로 보낸다.
- 질문이 너무 짧거나(“추천해줘”, “맛집”, “음식”) 지역/동/의도가 불명확하면 ambiguous로 보낸다. 단, “명동 추천”처럼 동이 명시된 경우에는 ambiguous가 아니라 in_scope로 두고 바로 답변 가능.

# 출력 형식(엄격)
반드시 JSON만 출력한다. 코드블록/설명/추가 텍스트 금지.

스키마:
{
  "route": "blocked" | "out_of_scope" | "ambiguous" | "in_scope",
  "allowed": true | false,
  "needs_soft_tone": true | false,
  "answerable": true | false,
  "message": string,
  "clarify_question": string | null
}

작성 규칙:
- blocked:
  - allowed=false, answerable=false
  - message: "부적절한 내용이 포함되어 있어 답변할 수 없어요. 표현을 바꿔 다시 질문해 주세요."
  - clarify_question=null
- out_of_scope:
  - allowed=true
  - answerable=false
  - message: "죄송하지만, 현재 시스템은 ‘서울 중구 8개 동 가이드북’ 범위에서만 답변할 수 있어요. 중구 기준으로 질문을 바꿔 주세요."
  - clarify_question=null
- ambiguous:
  - allowed=true
  - answerable=false
  - message: 간단 안내 1문장
  - clarify_question: 사용자가 바로 답할 수 있는 되묻기 1문장(선택지 포함 권장)
- in_scope:
  - allowed=true
  - answerable=true
  - message는 빈 문자열("")로 둔다
  - clarify_question=null

# 예시(그대로 따라하지 말고 판단 기준만 참고)
입력: "씨발" → blocked
입력: "ㅅㅂ" → blocked
입력: "시1발ㅋㅋ" → blocked
입력: "병신" → blocked
입력: "너 멍청하냐" → blocked
입력: "부산 해운대 맛집" → out_of_scope
입력: "명동 가볼만한 곳 추천" → in_scope
입력: "맛집 추천" → ambiguous (어느 동인지/어떤 종류인지 되묻기)
"""

def _safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        # JSON 외 텍스트 섞여 나오는 경우를 대비한 최소 방어
        # 가장 바깥 {...} 구간만 잘라서 파싱 시도
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
        raise


def judge_question(question: str) -> JudgeResult:
    """
    규칙 기반 없이 LLM이:
    - 차단 여부
    - 중구 범위 여부(out_of_scope)
    - 애매함(ambiguous) 여부 + 되묻기 질문
    - 문서로 답 가능(answerable) 여부
    를 판단한다.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    resp = llm.invoke([
        ("system", _JUDGE_SYSTEM),
        ("user", f"사용자 입력: {question}")
    ])

    data = _safe_json_loads(resp.content)

    route: Route = data.get("route", "ambiguous")
    allowed = bool(data.get("allowed", True))
    needs_soft_tone = bool(data.get("needs_soft_tone", False))
    answerable = bool(data.get("answerable", True))
    message = str(data.get("message", "")).strip()
    clarify_q = data.get("clarify_question", None)

    # 최소 보정
    if route == "blocked":
        allowed = False
        answerable = False

    if route == "ambiguous" and (clarify_q is None or str(clarify_q).strip() == ""):
        clarify_q = "어느 동(명동/을지로/회현 등) 중심으로, 어떤 종류(음식/카페/볼거리)로 찾고 계신가요?"

    if not message:
        # 안전한 기본 문구
        if route == "blocked":
            message = "부적절한 내용이 포함되어 있어 답변할 수 없어요. 표현을 바꿔 다시 질문해 주세요."
        elif route == "out_of_scope":
            message = "죄송하지만, 현재 시스템은 ‘서울 중구 8개 동 가이드북’ 범위에서만 답변할 수 있어요. 중구 기준으로 질문을 바꿔 주세요."
        elif route == "ambiguous":
            message = clarify_q
        else:
            # in_scope인데 answerable=false인 경우 대비
            message = "가능하면 더 구체적으로 질문해 주세요."

    return JudgeResult(
        route=route,
        allowed=allowed,
        needs_soft_tone=needs_soft_tone,
        answerable=answerable,
        message=message,
        clarify_question=clarify_q if route == "ambiguous" else None
    )