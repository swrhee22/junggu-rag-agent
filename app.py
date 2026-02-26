from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from agent.graph import run_agent

st.set_page_config(page_title="중구 가이드 Agent")

st.title("📍 중구 가이드 AI")

question = st.chat_input("질문을 입력하세요")

if question:
    with st.chat_message("user"):
        st.write(question)

    answer, docs = run_agent(question)

    with st.chat_message("assistant"):
        st.write(answer)

        st.markdown("### 🔎 참고 문서")
        for d in docs:
            st.write(f"- {d.metadata.get('source_file')} (page {d.metadata.get('page')})")