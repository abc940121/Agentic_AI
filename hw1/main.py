import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment Setup
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
BASE_URL = None

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ---------------------------------------------------------------------------
# Mock Data Functions
# ---------------------------------------------------------------------------

EXCHANGE_RATES = {
    "USD_TWD": "32.0",
    "JPY_TWD": "0.2",
    "EUR_USD": "1.2",
}

STOCK_PRICES = {
    "AAPL": "260.00",
    "TSLA": "430.00",
    "NVDA": "190.00",
}


def get_exchange_rate(currency_pair: str) -> str:
    """Return the exchange rate for a given currency pair as a JSON string."""
    rate = EXCHANGE_RATES.get(currency_pair.upper())
    if rate is None:
        return json.dumps({"error": "Data not found"})
    return json.dumps({"currency_pair": currency_pair.upper(), "rate": rate})


def get_stock_price(symbol: str) -> str:
    """Return the current stock price for a given symbol as a JSON string."""
    price = STOCK_PRICES.get(symbol.upper())
    if price is None:
        return json.dumps({"error": "Data not found"})
    return json.dumps({"symbol": symbol.upper(), "price": price})


# ---------------------------------------------------------------------------
# Function Map  (no if-else chains — dispatch via dictionary)
# ---------------------------------------------------------------------------

available_functions = {
    "get_exchange_rate": get_exchange_rate,
    "get_stock_price": get_stock_price,
}

# ---------------------------------------------------------------------------
# Tool Schemas  (OpenAI format, strict mode enabled)
# ---------------------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": (
                "Retrieve the exchange rate for a given currency pair. "
                "The pair must be in the format 'BASE_QUOTE', e.g. 'USD_TWD'."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "currency_pair": {
                        "type": "string",
                        "description": (
                            "Currency pair in BASE_QUOTE format, e.g. 'USD_TWD', "
                            "'JPY_TWD', 'EUR_USD'."
                        ),
                    }
                },
                "required": ["currency_pair"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Retrieve the current stock price for a given ticker symbol, "
                "e.g. 'AAPL', 'TSLA', 'NVDA'."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'NVDA'.",
                    }
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable and professional Financial Assistant.
Your capabilities include:
- Looking up real-time stock prices for supported ticker symbols (AAPL, TSLA, NVDA).
- Retrieving exchange rates for supported currency pairs (USD_TWD, JPY_TWD, EUR_USD).
- Answering general financial questions and maintaining context across the conversation.

Always be concise, accurate, and helpful. When data is unavailable, politely inform the user."""

# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------

def run_agent():
    print("=" * 60)
    print("  Financial Assistant  (type 'exit' or 'quit' to stop)")
    print("=" * 60)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        # --- Get user input ---
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Goodbye! Have a great day.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye! Have a great day.")
            break

        messages.append({"role": "user", "content": user_input})

        # --- Agentic loop: keep calling LLM until no more tool calls ---
        while True:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            response_message = response.choices[0].message

            # Append assistant turn to history (preserves tool_calls if any)
            messages.append(response_message)

            # If the model wants to call tools, execute ALL of them in parallel
            if response_message.tool_calls:
                print("\n[DEBUG] Tool calls requested:")
                for tc in response_message.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    print(f"  → {fn_name}({fn_args})")

                    # Dispatch via Function Map
                    fn_result = available_functions[fn_name](**fn_args)
                    print(f"    Result: {fn_result}")

                    # Append each tool result to history
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": fn_result,
                        }
                    )

                # Continue the loop so the LLM can formulate its final answer
                continue

            # No tool calls — this is the final text response
            final_answer = response_message.content or ""
            print(f"\nAssistant: {final_answer}")
            break


if __name__ == "__main__":
    run_agent()
