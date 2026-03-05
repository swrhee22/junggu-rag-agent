# Junggu Guide AI Agent

서울 중구 가이드북(8개 동)을 기반으로 질문에 답하는 RAG 기반 챗봇 에이전트 프로젝트.

본 프로젝트는 단순 Retrieval-Augmented Generation 구조에서 시작하여
LangGraph 기반 Agent 아키텍처로 확장하였다.

---

# Project Goal

본 프로젝트의 목표는 다음과 같다.

- PDF 기반 RAG 시스템 구축
- LangGraph 기반 Agent 구조 설계
- LLM 기반 Guard 및 Routing 구현
- Context-aware Retrieval 구현
- Structured LLM Output + Validation 적용

---

# System Architecture

시스템은 다음 두 단계로 구성된다.

1. Offline Pipeline (문서 인덱싱)
2. Online Pipeline (질문 응답 Agent)

---

# Offline Pipeline

PDF 문서를 벡터 데이터베이스로 변환하는 과정

1. PDF 문서 로드
2. 문서 Chunk 분할
3. OpenAI Embedding 생성
4. Chroma Vector Database 저장

---

# Online Pipeline

사용자가 질문하면 Agent 시스템이 다음 과정을 수행한다.

1. 사용자 질문 수신
2. 입력 Guard 및 Routing 판단
3. 질문 범위 판단 (중구 관련 여부)
4. Clarify 질문 여부 판단
5. Vector DB 검색
6. LLM 답변 생성
7. 결과 반환

---

# Project Structure

```
project
│
├─ app.py
│   Streamlit Chat UI
│
├─ agent
│   ├─ graph.py
│   │   LangGraph orchestration
│   │
│   ├─ tools.py
│   │   retriever / judge / utility functions
│   │
│   └─ prompts.py
│       prompt templates
│
├─ ingest
│   └─ build_index.py
│       PDF indexing pipeline
│
├─ data
│   └─ pdf
│
└─ db
    └─ chroma
```

---

# Tech Stack

- Python
- LangChain
- LangGraph
- OpenAI
- Chroma VectorDB
- Streamlit
- Pydantic

---

# Run

## Index documents

python ingest/build_index.py

## Run chatbot

streamlit run app.py