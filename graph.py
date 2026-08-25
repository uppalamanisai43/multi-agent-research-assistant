from typing import TypedDict, List, Dict
import time

import streamlit as st
from groq import RateLimitError
from google import genai
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient


class ResearchState(TypedDict):
    topic: str
    depth: str
    provider: str
    plan: str
    queries: List[str]
    sources: List[Dict[str, str]]
    critic_feedback: str
    decision: str
    iteration_count: int
    report: str
    events: List[str]


groq_llm = ChatGroq(
    model="groq/compound-mini",
    api_key=st.secrets["GROQ_API_KEY"],
    max_tokens=500
)

gemini_client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

tavily = TavilyClient(
    api_key=st.secrets["TAVILY_API_KEY"]
)


def remove_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()

    return text.strip()


def ask_llm(prompt: str, provider: str) -> str:
    """Call the selected provider; retry Groq if temporarily rate-limited."""

    if provider == "Gemini":
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return remove_thinking(response.text or "")

    for attempt in range(3):
        try:
            return remove_thinking(groq_llm.invoke(prompt).content)

        except RateLimitError:
            if attempt == 2:
                raise

            time.sleep(8)

    raise RuntimeError("Unable to receive an LLM response.")


def planner(state: ResearchState):
    feedback = state.get("critic_feedback", "")

    prompt = f"""
You are a research planner.

Topic: {state["topic"]}
Research depth: {state["depth"]}
Critic feedback: {feedback or "No previous feedback."}

Create a short research plan and exactly three web-search queries.

For Quick, use direct broad queries.
For Standard, cover benefits, risks, and evidence.
For Deep, use detailed queries covering evidence, limitations, and policy context.

If critic feedback exists, target the missing information.

Use exactly this format:

PLAN:
<plan>

QUERIES:
- <query 1>
- <query 2>
- <query 3>
"""

    response = ask_llm(prompt, state["provider"])
    plan_part, query_part = response.split("QUERIES:", 1)

    queries = [
        line.lstrip("- ").strip()
        for line in query_part.splitlines()
        if line.strip().startswith("-")
    ]

    return {
        "plan": plan_part.replace("PLAN:", "").strip(),
        "queries": queries[:3],
        "events": state.get("events", []) + [
            "Planner created the research plan."
        ]
    }


def researcher(state: ResearchState):
    collected_sources = []

    for query in state["queries"]:
        response = tavily.search(
            query=query,
            search_depth="basic",
            max_results=2
        )

        for item in response.get("results", []):
            collected_sources.append({
                "title": item.get("title", "Untitled source"),
                "url": item.get("url", ""),
                "content": item.get("content", "")
            })

    unique_sources = []
    seen_urls = set()

    for source in collected_sources:
        if source["url"] and source["url"] not in seen_urls:
            unique_sources.append(source)
            seen_urls.add(source["url"])

    sources = unique_sources[:5]

    return {
        "sources": sources,
        "events": state["events"] + [
            f"Researcher collected {len(sources)} sources."
        ]
    }


def critic(state: ResearchState):
    source_text = "\n\n".join(
        f"Title: {source['title']}\n"
        f"URL: {source['url']}\n"
        f"Snippet: {source['content'][:250]}"
        for source in state["sources"]
    )

    prompt = f"""
You are a research critic.

Topic: {state["topic"]}
Research plan: {state["plan"]}

Review these sources. Use APPROVE if they are sufficient for a short factual
report; otherwise use RESEARCH_MORE.

Reply in exactly this format:

DECISION: APPROVE or RESEARCH_MORE
FEEDBACK: <brief feedback>

Sources:
{source_text}
"""

    response = ask_llm(prompt, state["provider"])
    decision = "RESEARCH_MORE" if "RESEARCH_MORE" in response else "APPROVE"

    return {
        "decision": decision,
        "critic_feedback": response.split("FEEDBACK:", 1)[-1].strip(),
        "iteration_count": state["iteration_count"] + 1,
        "events": state["events"] + [f"Critic decision: {decision}"]
    }


def writer(state: ResearchState):
    source_text = "\n\n".join(
        f"[{index}] {source['title']}\n"
        f"URL: {source['url']}\n"
        f"Content: {source['content'][:350]}"
        for index, source in enumerate(state["sources"], start=1)
    )

    prompt = f"""
You are a careful research report writer.

Topic: {state["topic"]}
Research depth: {state["depth"]}
Research plan: {state["plan"]}
Critic feedback: {state["critic_feedback"]}

Write a concise Markdown report with:

# Title
## Executive Summary
## Key Findings
## Limitations
## Sources

Rules:
- Use only the sources given below.
- Never invent facts or URLs.
- Cite claims in the report as [1], [2], and so on.
- Under ## Sources, use exactly this format:

- [1] Source title — https://example.com

- Never put citation labels such as [1] or 【1】 inside a URL.
- Do not show reasoning or think tags.
- For finance-related topics, include this statement under Limitations:
  "Educational research only — not financial advice."

Sources:
{source_text}
"""

    return {
        "report": ask_llm(prompt, state["provider"]),
        "events": state["events"] + ["Writer generated the final report."]
    }


def choose_next_step(state: ResearchState):
    if (
        state["decision"] == "RESEARCH_MORE"
        and state["iteration_count"] < 3
    ):
        return "planner"

    return "writer"


graph = StateGraph(ResearchState)

graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("critic", critic)
graph.add_node("writer", writer)

graph.set_entry_point("planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "critic")

graph.add_conditional_edges(
    "critic",
    choose_next_step,
    {
        "planner": "planner",
        "writer": "writer"
    }
)

graph.add_edge("writer", END)

research_app = graph.compile()
