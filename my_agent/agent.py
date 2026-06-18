from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search

root_agent = Agent(
    model='gemini-2.5-flash',
    name='root_agent',
    description='A helpful assistant for user questions.',
    instruction='You are a helpful assistant. Use Google Search for current info or if unsure.',
    tools=[google_search],
)
