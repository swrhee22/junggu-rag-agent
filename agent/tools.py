# agent/tools.py
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# .env 로드 (Streamlit에서 실행해도 환경변수 읽히게)
load_dotenv()


# ====== 1) Retrieval (Chroma) ======
DEFAULT_DB_DIR = "db/chroma"
DEFAULT_COLLECTION = "junggu_guides"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def get_embeddings(model: str = DEFAULT_EMBED_MODEL) -> OpenAIEmbeddings:
    """
    OpenAI 임베딩 객체 생성.
    OPENAI_API_KEY는 .env 또는 환경변수에 있어야 함.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. "
            "프로젝트 루트의 .env에 OPENAI_API_KEY=... 형태로 넣어주세요."
        )
    return OpenAIEmbeddings(model=model)


def get_vectorstore(
    db_dir: str = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> Chroma:
    """
    로컬 디스크에 저장된 Chroma DB를 로드.
    build_index.py에서 같은 collection_name/persist_directory로 저장했어야 함.
    """
    embeddings = get_embeddings()
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=db_dir,
        embedding_function=embeddings,  # 최신 방식
    )
    return vs


def get_retriever(
    k: int = 4,
    db_dir: str = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
):
    """
    RAG에서 사용할 retriever.
    k = 검색해서 가져올 chunk 개수
    """
    vs = get_vectorstore(db_dir=db_dir, collection_name=collection_name)
    return vs.as_retriever(search_kwargs={"k": k})


# ====== 2) Guardrail (룰 기반 1차) ======
@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""


# 너무 빡세게 막지 말고, "명백한" 욕설/혐오/성적 표현 정도만 1차 차단
_BAD_PATTERNS: List[str] = [
    r"\b씨발\b",
    r"\b시발\b",
    r"\b병신\b",
    r"\b좆\b",
    r"\bㅅㅂ\b",
    r"\bㅂㅅ\b",
    r"\b좆같\b",
    r"\b개새끼\b",
]


def guard_user_input(text: str) -> GuardResult:
    """
    사용자 입력이 부적절하면 차단.
    (나중에 LLM 기반 moderation으로 고도화 가능)
    """
    t = (text or "").strip().lower()
    if not t:
        return GuardResult(False, "질문이 비어있어요. 내용을 입력해 주세요.")

    for pat in _BAD_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return GuardResult(False, "부적절한 표현이 포함되어 있어 답변할 수 없어요. 표현을 바꿔서 질문해 주세요.")

    return GuardResult(True, "")


# ====== 3) Scope Router (룰 기반 1차) ======
JUNGGU_HINTS = [
    "중구", "명동", "을지로", "회현", "필동", "장충동", "소공동", "광희동", "중림동"
]

# 자주 나오는 “중구가 아닌” 지역 키워드 몇 개만(확장 가능)
OUT_OF_SCOPE_HINTS = [
    "부산", "해운대", "서면", "광안리",
    "대구", "동성로",
    "광주", "전주", "여수",
    "제주",
    "강남구", "서초구", "송파구", "마포구", "용산구", "성동구", "종로구", "강북구",
]


def is_about_junggu(text: str) -> bool:
    t = (text or "").strip()
    return any(h in t for h in JUNGGU_HINTS)


def is_out_of_scope(text: str) -> bool:
    """
    중구 관련 키워드가 없고, 다른 지역 힌트가 있으면 out-of-scope로 판단(룰 기반).
    """
    t = (text or "").strip()
    if is_about_junggu(t):
        return False
    return any(h in t for h in OUT_OF_SCOPE_HINTS)

def format_sources(docs) -> list[str]:
    """
    Chroma에서 나온 문서 chunk들의 metadata를 바탕으로
    '파일명 (page n)' 형태의 출처 리스트 생성
    """
    sources = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        file_name = meta.get("source_file", "unknown.pdf")
        page = meta.get("page", None)

        if page is not None:
            sources.append(f"{file_name} (page {page})")
        else:
            sources.append(file_name)

    # 중복 제거 (순서 유지)
    seen = set()
    unique = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return unique