"""Day 1b: Multi-Agent Systems & Workflow Patterns.

Standalone re-implementation of the Kaggle 5-Day Agents course notebook,
covering four ways to coordinate multiple ADK agents:

1. LLM Orchestrator  - root agent calls sub-agents wrapped as AgentTools
2. SequentialAgent   - fixed pipeline, output_key chains state between steps
3. ParallelAgent     - independent sub-agents run concurrently, then aggregate
4. LoopAgent         - Critic/Refiner cycle until "APPROVED" or max_iterations

Run with: python multi_agent_lab.py
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent, ParallelAgent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(__file__), "my_agent", ".env"))

MODEL = "gemini-2.5-flash-lite"
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)


def gemini():
    return Gemini(model=MODEL, retry_options=RETRY_CONFIG)


async def section_section_header(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Section 2: LLM Orchestrator - Research & Summarization
# ---------------------------------------------------------------------------
async def run_llm_orchestrator():
    await section_section_header(
        "SECTION 2: LLM Orchestrator (Research -> Summarize)"
    )

    research_agent = Agent(
        name="ResearchAgent",
        model=gemini(),
        instruction="""You are a specialized research agent. Your only job is to use the
        google_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
        tools=[google_search],
        output_key="research_findings",
    )

    summarizer_agent = Agent(
        name="SummarizerAgent",
        model=gemini(),
        instruction="""Read the provided research findings: {research_findings}
Create a concise summary as a bulleted list with 3-5 key points.""",
        output_key="final_summary",
    )

    root_agent = Agent(
        name="ResearchCoordinator",
        model=gemini(),
        instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
1. First, you MUST call the `ResearchAgent` tool to find relevant information on the topic provided by the user.
2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` tool to create a concise summary.
3. Finally, present the final summary clearly to the user as your response.""",
        tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
    )

    runner = InMemoryRunner(agent=root_agent)
    await runner.run_debug(
        "What are the latest advancements in quantum computing and what do they mean for AI?"
    )


# ---------------------------------------------------------------------------
# Section 3: SequentialAgent - Blog Post Pipeline
# ---------------------------------------------------------------------------
async def run_sequential_pipeline():
    await section_section_header(
        "SECTION 3: SequentialAgent (Outline -> Write -> Edit)"
    )

    outline_agent = Agent(
        name="OutlineAgent",
        model=gemini(),
        instruction="""Create a blog outline for the given topic with:
        1. A catchy headline
        2. An introduction hook
        3. 3-5 main sections with 2-3 bullet points for each
        4. A concluding thought""",
        output_key="blog_outline",
    )

    writer_agent = Agent(
        name="WriterAgent",
        model=gemini(),
        instruction="""Following this outline strictly: {blog_outline}
        Write a brief, 200 to 300-word blog post with an engaging and informative tone.""",
        output_key="blog_draft",
    )

    editor_agent = Agent(
        name="EditorAgent",
        model=gemini(),
        instruction="""Edit this draft: {blog_draft}
        Your task is to polish the text by fixing any grammatical errors, improving the flow and sentence structure, and enhancing overall clarity.""",
        output_key="final_blog",
    )

    root_agent = SequentialAgent(
        name="BlogPipeline",
        sub_agents=[outline_agent, writer_agent, editor_agent],
    )

    runner = InMemoryRunner(agent=root_agent)
    await runner.run_debug(
        "Write a blog post about the benefits of multi-agent systems for software developers"
    )


# ---------------------------------------------------------------------------
# Section 4: ParallelAgent - Multi-Topic Research
# ---------------------------------------------------------------------------
async def run_parallel_research():
    await section_section_header(
        "SECTION 4: ParallelAgent (Tech + Health + Finance -> Aggregate)"
    )

    tech_researcher = Agent(
        name="TechResearcher",
        model=gemini(),
        instruction="""Research the latest AI/ML trends. Include 3 key developments,
the main companies involved, and the potential impact. Keep the report very concise (100 words).""",
        tools=[google_search],
        output_key="tech_research",
    )

    health_researcher = Agent(
        name="HealthResearcher",
        model=gemini(),
        instruction="""Research recent medical breakthroughs. Include 3 significant advances,
their practical applications, and estimated timelines. Keep the report concise (100 words).""",
        tools=[google_search],
        output_key="health_research",
    )

    finance_researcher = Agent(
        name="FinanceResearcher",
        model=gemini(),
        instruction="""Research current fintech trends. Include 3 key trends,
their market implications, and the future outlook. Keep the report concise (100 words).""",
        tools=[google_search],
        output_key="finance_research",
    )

    aggregator_agent = Agent(
        name="AggregatorAgent",
        model=gemini(),
        instruction="""Combine these three research findings into a single executive summary:

        **Technology Trends:**
        {tech_research}

        **Health Breakthroughs:**
        {health_research}

        **Finance Innovations:**
        {finance_research}

        Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words.""",
        output_key="executive_summary",
    )

    parallel_research_team = ParallelAgent(
        name="ParallelResearchTeam",
        sub_agents=[tech_researcher, health_researcher, finance_researcher],
    )

    root_agent = SequentialAgent(
        name="ResearchSystem",
        sub_agents=[parallel_research_team, aggregator_agent],
    )

    runner = InMemoryRunner(agent=root_agent)
    await runner.run_debug(
        "Run the daily executive briefing on Tech, Health, and Finance"
    )


# ---------------------------------------------------------------------------
# Section 5: LoopAgent - Story Refinement
# ---------------------------------------------------------------------------
def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}


async def run_loop_refinement():
    await section_section_header(
        "SECTION 5: LoopAgent (Write -> Critique -> Refine until APPROVED)"
    )

    initial_writer_agent = Agent(
        name="InitialWriterAgent",
        model=gemini(),
        instruction="""Based on the user's prompt, write the first draft of a short story (around 100-150 words).
        Output only the story text, with no introduction or explanation.""",
        output_key="current_story",
    )

    critic_agent = Agent(
        name="CriticAgent",
        model=gemini(),
        instruction="""You are a constructive story critic. Review the story provided below.
        Story: {current_story}

        Evaluate the story's plot, characters, and pacing.
        - If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
        - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
        output_key="critique",
    )

    refiner_agent = Agent(
        name="RefinerAgent",
        model=gemini(),
        instruction="""You are a story refiner. You have a story draft and critique.

        Story Draft: {current_story}
        Critique: {critique}

        Your task is to analyze the critique.
        - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
        - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.""",
        output_key="current_story",
        tools=[FunctionTool(exit_loop)],
    )

    story_refinement_loop = LoopAgent(
        name="StoryRefinementLoop",
        sub_agents=[critic_agent, refiner_agent],
        max_iterations=2,
    )

    root_agent = SequentialAgent(
        name="StoryPipeline",
        sub_agents=[initial_writer_agent, story_refinement_loop],
    )

    runner = InMemoryRunner(agent=root_agent)
    await runner.run_debug(
        "Write a short story about a lighthouse keeper who discovers a mysterious, glowing map"
    )


async def main():
    await run_llm_orchestrator()
    await run_sequential_pipeline()
    await run_parallel_research()
    await run_loop_refinement()


if __name__ == "__main__":
    asyncio.run(main())
