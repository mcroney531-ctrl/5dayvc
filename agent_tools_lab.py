"""Day 2a: Agent Tools.

Standalone re-implementation of the Kaggle 5-Day Agents course notebook,
covering how to extend agents with custom tools:

1. Custom function tools - plain Python functions exposed to an agent
2. Reliability via code execution + delegation - BuiltInCodeExecutor for
   math, and an AgentTool specialist so the root agent never does
   arithmetic itself

Run with: python agent_tools_lab.py
"""

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(__file__), "my_agent", ".env"))

MODEL = "gemini-2.5-flash-lite"
RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

PAYMENT_METHOD_FEES = {
    "credit_card": 0.029,
    "bank_transfer": 0.005,
    "crypto": 0.01,
}

EXCHANGE_RATES = {
    ("USD", "EUR"): 0.92,
    ("USD", "GBP"): 0.79,
    ("USD", "JPY"): 156.5,
    ("EUR", "USD"): 1.09,
    ("GBP", "USD"): 1.27,
}


def gemini():
    return Gemini(model=MODEL, retry_options=RETRY_CONFIG)


async def print_section_header(title: str):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# Section 2: Custom Tools - Currency Conversion
# ---------------------------------------------------------------------------
def get_fee_for_payment_method(method: str) -> dict:
    """Look up the processing fee percentage for a payment method.

    Args:
        method: One of "credit_card", "bank_transfer", or "crypto".

    Returns:
        A dict with status "success" and fee_percentage, or status "error"
        and an error_message if the method is not recognized.
    """
    fee = PAYMENT_METHOD_FEES.get(method.lower())
    if fee is None:
        return {
            "status": "error",
            "error_message": f"Unknown payment method: {method}",
        }
    return {"status": "success", "fee_percentage": fee}


def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Look up the exchange rate between two currencies.

    Args:
        base_currency: The 3-letter currency code to convert from.
        target_currency: The 3-letter currency code to convert to.

    Returns:
        A dict with status "success" and rate, or status "error" and an
        error_message if the currency pair is not supported.
    """
    rate = EXCHANGE_RATES.get((base_currency.upper(), target_currency.upper()))
    if rate is None:
        return {
            "status": "error",
            "error_message": f"No rate available for {base_currency} -> {target_currency}",
        }
    return {"status": "success", "rate": rate}


async def run_custom_tools():
    await print_section_header("SECTION 2: Custom Tools (Currency Conversion)")

    currency_agent = Agent(
        name="currency_agent",
        model=gemini(),
        instruction="""You are a smart currency conversion assistant.
1. Use `get_fee_for_payment_method` to look up the fee for the user's payment method.
2. Use `get_exchange_rate` to look up the conversion rate for the requested currencies.
3. If either tool returns an error, tell the user clearly what went wrong.
4. Otherwise, calculate the final converted amount after the fee is deducted and present it clearly.""",
        tools=[get_fee_for_payment_method, get_exchange_rate],
    )

    runner = InMemoryRunner(agent=currency_agent)
    await runner.run_debug(
        "I want to convert 500 USD to EUR, paying with a credit card. How much will I receive?"
    )


# ---------------------------------------------------------------------------
# Section 3: Reliability - Code Execution & Delegation
# ---------------------------------------------------------------------------
async def run_reliable_delegation():
    await print_section_header(
        "SECTION 3: Reliability (Code Execution + Specialist Delegation)"
    )

    calculation_agent = Agent(
        name="calculation_agent",
        model=gemini(),
        instruction="""You are a specialized calculator. ONLY respond by writing and
executing Python code to compute the requested value. Do not perform arithmetic yourself
in natural language - always rely on the code execution result.""",
        code_executor=BuiltInCodeExecutor(),
    )

    enhanced_currency_agent = Agent(
        name="enhanced_currency_agent",
        model=gemini(),
        instruction="""You are a smart currency conversion assistant.
1. Use `get_fee_for_payment_method` to look up the fee for the user's payment method.
2. Use `get_exchange_rate` to look up the conversion rate for the requested currencies.
3. If either tool returns an error, tell the user clearly what went wrong.
4. You MUST delegate all arithmetic (applying the fee, multiplying by the exchange rate) to
   the `calculation_agent` tool. NEVER compute the final amount yourself.
5. Present the final converted amount clearly to the user.""",
        tools=[
            get_fee_for_payment_method,
            get_exchange_rate,
            AgentTool(calculation_agent),
        ],
    )

    runner = InMemoryRunner(agent=enhanced_currency_agent)
    await runner.run_debug(
        "I want to convert 1250 USD to GBP, paying with a bank transfer. How much will I receive?"
    )


async def main():
    await run_custom_tools()
    await run_reliable_delegation()


if __name__ == "__main__":
    asyncio.run(main())
