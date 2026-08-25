# Multi-Agent Research Assistant

A Streamlit application that uses a LangGraph multi-agent workflow to research a topic and generate a cited Markdown report.

## Agents

1. Planner — creates research queries
2. Researcher — searches live web sources using Tavily
3. Critic — reviews source quality and requests more research when needed
4. Writer — creates the final cited report

## Features

- Quick, Standard, and Deep research modes
- Tavily live web search
- Up to 3 feedback-loop iterations
- Live agent execution status
- Groq rate-limit retry handling
- Markdown report download
- Finance disclaimer for educational use

## Tech Stack

- Python
- Streamlit
- LangGraph
- Groq
- Tavily

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
