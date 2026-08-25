import streamlit as st
from graph import research_app

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Multi-Agent Research Assistant")
st.caption("Planner → Researcher → Critic → Writer")

st.info(
    "Educational research only — not financial advice or a recommendation "
    "to buy, sell, or invest."
)

depth = st.selectbox(
    "Research depth",
    ["Quick", "Standard", "Deep"],
    index=1
)

provider = st.selectbox(
    "LLM provider",
    ["Groq", "Gemini"]
)

topic = st.text_area(
    "Enter a research topic",
    placeholder="Example: Benefits and risks of artificial intelligence in education",
    height=120
)

if st.button("Generate Report", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        initial_state = {
            "topic": topic,
            "depth": depth,
            "provider": provider,
            "plan": "",
            "queries": [],
            "sources": [],
            "critic_feedback": "",
            "decision": "",
            "iteration_count": 0,
            "report": "",
            "events": []
        }

        last_event = ""
        final_result = None

        with st.status("Research agents are working...", expanded=True) as status:
            for state in research_app.stream(
                initial_state,
                stream_mode="values"
            ):
                final_result = state

                if state["events"]:
                    event = state["events"][-1]

                    if event != last_event:
                        status.write(event)
                        last_event = event

            status.update(
                label="Research complete",
                state="complete",
                expanded=False
            )

        st.divider()
        st.markdown(final_result["report"])

        st.download_button(
            label="Download Markdown Report",
            data=final_result["report"],
            file_name="research_report.md",
            mime="text/markdown"
        )
