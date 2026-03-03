import streamlit as st
from agent.graph import run_agent

st.set_page_config(page_title="중구 가이드 AI", page_icon="📍", layout="centered")

st.title("📍 중구 가이드 AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None  

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("🔎 참고 문서"):
                for s in m["sources"]:
                    st.write(f"- {s}")
        if m.get("route"):
            st.caption(f"route: `{m['route']}`")


question = st.chat_input("질문을 입력하세요")

if question:
    if st.session_state.pending_question:
        merged = (
            f"{st.session_state.pending_question}\n\n"
            f"(추가정보/후속답변) {question}"
        )
        user_visible = question 
        final_question = merged  
        st.session_state.pending_question = None
    else:
        user_visible = question
        final_question = question

    st.session_state.messages.append({"role": "user", "content": user_visible})
    with st.chat_message("user"):
        st.markdown(user_visible)

    with st.chat_message("assistant"):
        with st.spinner("생각하는 중..."):
            answer, sources, route = run_agent(final_question)

        st.markdown(answer)

        if sources:
            with st.expander("🔎 참고 문서"):
                for s in sources:
                    st.write(f"- {s}")

        st.caption(f"route: `{route}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "route": route}
    )

    if route == "ambiguous":
        st.session_state.pending_question = user_visible