import streamlit as st
from agent.graph import run_agent

st.set_page_config(page_title="중구 가이드 AI", page_icon="📍", layout="centered")

st.title("📍 중구 가이드 AI")

# 세션에 대화 저장
if "messages" not in st.session_state:
    st.session_state.messages = []

# ambiguous(되묻기) 이후 후속 답변을 합치기 위한 보관
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None  # 직전 사용자의 원 질문
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None    # assistant가 되물었던 문장(옵션)

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
    # 1) ambiguous 후속 답변 합치기:
    # pending_question이 있으면 (직전 질문 + 이번 답변)을 합쳐서 처리
    if st.session_state.pending_question:
        merged = (
            f"{st.session_state.pending_question}\n\n"
            f"(추가정보/후속답변) {question}"
        )
        user_visible = question  # 화면에는 사용자가 입력한 그대로 보여주고
        final_question = merged  # 모델에는 합친 질문을 던짐
        # pending은 이번 턴에서 소모(다시 ambiguous면 아래에서 재설정)
        st.session_state.pending_question = None
        st.session_state.pending_prompt = None
    else:
        user_visible = question
        final_question = question

    # 유저 메시지 저장/렌더 (화면에는 그대로)
    st.session_state.messages.append({"role": "user", "content": user_visible})
    with st.chat_message("user"):
        st.markdown(user_visible)

    # 답변 생성 (스피너)
    with st.chat_message("assistant"):
        with st.spinner("생각하는 중..."):
            answer, sources, route = run_agent(final_question)

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

    # 2) route가 ambiguous면, "이번 턴의 원 질문"을 pending으로 저장
    # - 다음 턴에 사용자가 "음식" 같은 후속 답을 하면 자동 합치기
    if route == "ambiguous":
        # pending_question은 "사용자의 직전 원 질문"이어야 UX가 자연스러움
        # final_question이 아니라 user_visible을 저장하는게 핵심
        st.session_state.pending_question = user_visible