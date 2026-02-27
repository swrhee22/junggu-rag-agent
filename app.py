import streamlit as st
from agent.graph import run_agent

st.set_page_config(page_title="중구 가이드 AI", page_icon="📍", layout="centered")

st.title("📍 중구 가이드 AI")
# 세션에 대화 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 렌더
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("🔎 참고 문서"):
                for s in m["sources"]:
                    st.write(f"- {s}")
        if m.get("route"):
            st.caption(f"route: `{m['route']}`")

# 입력
question = st.chat_input("질문을 입력하세요")

if question:
    # 유저 메시지
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 답변 생성 (스피너)
    with st.chat_message("assistant"):
        with st.spinner("생각하는 중..."):
            answer, sources, route = run_agent(question)

        st.markdown(answer)

        # 참고 문서
        if sources:
            with st.expander("🔎 참고 문서"):
                for s in sources:
                    st.write(f"- {s}")

        st.caption(f"route: `{route}`")

    # 세션 저장
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "route": route}
    )