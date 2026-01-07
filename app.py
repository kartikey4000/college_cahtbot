import streamlit as st
from chatbot import answer

st.set_page_config(page_title="College Chatbot")

st.title("🎓 College Information Chatbot")

query = st.text_input("Ask anything about the college")

if query:
    with st.spinner("Searching..."):
        reply, sources = answer(query)

    st.markdown("### 🤖 Answer")
    st.write(reply)

    st.markdown("### 📚 Sources")
    for s in sources:
        st.write("-", s)
