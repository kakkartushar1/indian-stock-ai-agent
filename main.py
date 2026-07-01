"""
Indian Stock Analyst Agent - Main Entry Point

This is the main entry point for the Indian Stock Analyst Agent.
Run this file to start analyzing stocks.

Usage:
    python main.py                    # Interactive mode
    python main.py "Analyze RELIANCE" # Single query mode
    python main.py --help             # Show help
"""

import asyncio
import sys
import os

# Ensure the project directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import (
    stock_analyst_agent,
    multi_agent_analyst,
    analyze_stock,
    analyze_stock_sync,
    interactive_session,
)
from agents import Runner
from agents.pipeline_orchestrator import run_full_analysis_with_pdf
from config import MAX_TURNS
from llm_provider import get_configuration_error, get_provider_status, is_llm_configured
import re

# Use multi-agent system by default
USE_MULTI_AGENT = True
# Use pipeline mode for guaranteed agent execution
USE_PIPELINE_MODE = True


def print_banner():
    """Print the application banner."""
    banner = """
    ================================================================
    |                                                              |
    |         INDIAN STOCK ANALYST AI (10-Agent System)           |
    |                                                              |
    |   Powered by OpenAI Agents SDK (Multi-Agent Architecture)   |
    |   Analyze NSE & BSE Stocks with AI                          |
    |                                                              |
    |   Agents: Fundamental, Technical, Sentiment, Macro,         |
    |           Document, Bull, Bear, Judge, Risk, Portfolio      |
    |                                                              |
    ================================================================
    """
    print(banner)


def print_help():
    """Print help information."""
    help_text = """
    USAGE:
        python main.py                      Start interactive session
        python main.py "<query>"            Analyze with single query
        python main.py --help               Show this help message

    EXAMPLES:
        python main.py "Should I buy RELIANCE?"
        python main.py "Analyze TCS stock for investment"
        python main.py "I own INFY at 1500, should I hold or sell?"
        python main.py "Compare HDFCBANK and ICICIBANK"

    SUPPORTED STOCKS:
        - All NSE stocks (use symbol like RELIANCE, TCS, INFY)
        - All BSE stocks (add .BO suffix like RELIANCE.BO)
        - Indian indices (NIFTY50, SENSEX)

    FEATURES:
        - Multi-Agent Architecture (Fundamental, Technical, Sentiment)
        - Real-time price data from NSE/BSE
        - Fundamental analysis (P/E, P/B, ROE, Debt, Growth)
        - Technical analysis (RSI, MACD, Moving Averages, Trends)
        - Support & Resistance levels with Fibonacci
        - News sentiment analysis with VADER + TextBlob
        - Bull/Bear case synthesis
        - Professional PDF report generation

    ENVIRONMENT SETUP:
        Set your LLM provider and API key:
        LLM_PROVIDER=openai
        OPENAI_API_KEY=your-api-key-here

        Or create a .env file:
        LLM_PROVIDER=groq
        GROQ_API_KEY=your-groq-key
    """
    print(help_text)


def extract_stock_symbol(query: str) -> str:
    """Extract stock symbol from query."""
    # Common patterns for stock symbols
    query_upper = query.upper()

    # List of common Indian stocks and ETFs
    common_stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
        "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
        "SUNPHARMA", "BAJFINANCE", "WIPRO", "HCLTECH", "ULTRACEMCO",
        "TATAMOTORS", "TATASTEEL", "ONGC", "NTPC", "POWERGRID",
        "ADANIENT", "ADANIPORTS", "TECHM", "NESTLEIND", "BRITANNIA",
        # Popular ETFs
        "GROWWSLVR", "SILVERBEES", "GOLDBEES", "NIFTYBEES", "BANKBEES",
        "SILVERETF", "SILVER", "GOLDETF", "LIQUIDBEES"
    ]

    # Check for common stocks in query
    for stock in common_stocks:
        if stock in query_upper:
            return stock

    # Try to find pattern like "SYMBOL stock" or "SYMBOL.NS"
    patterns = [
        r'\b([A-Z]{2,15})\.NS\b',
        r'\b([A-Z]{2,15})\.BO\b',
        r'\b([A-Z]{2,15})\s+stock\b',
        r'analyze\s+([A-Z]{2,15})\b',
        r'buy\s+([A-Z]{2,15})\b',
        r'sell\s+([A-Z]{2,15})\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, query_upper)
        if match:
            return match.group(1)

    # Default: try to find any uppercase word that looks like a stock symbol
    words = query_upper.split()
    for word in words:
        if word.isalpha() and 2 <= len(word) <= 15 and word not in ['STOCK', 'ANALYSIS', 'BUY', 'SELL', 'HOLD', 'THE', 'AND', 'FOR']:
            return word

    return "RELIANCE"  # Default fallback


async def single_query_mode(query: str):
    """Run a single query and print the result."""
    print(f"\n[Analyzing]: {query}\n")
    print("-" * 50)

    # Use pipeline mode for guaranteed agent execution
    if USE_PIPELINE_MODE:
        print("[Pipeline Mode]: Running ALL 9 agents sequentially...\n")

        try:
            # Extract stock symbol from query
            stock_symbol = extract_stock_symbol(query)
            print(f"[Detected Stock]: {stock_symbol}")

            # Run the pipeline
            result = await run_full_analysis_with_pdf(stock_symbol, query)
            # Handle Unicode for Windows console
            try:
                print(f"\n{result}")
            except UnicodeEncodeError:
                # Replace problematic characters for Windows console
                safe_result = result.encode('ascii', 'replace').decode('ascii')
                print(f"\n{safe_result}")

        except Exception as e:
            print(f"\n[ERROR]: {str(e)}")
            import traceback
            traceback.print_exc()
            print("\nPlease check:")
            print("1. Your LLM_PROVIDER and provider API key are set correctly")
            print("2. You have internet connectivity")
            print("3. The stock symbol is valid")

    else:
        # Legacy mode using handoffs
        agent = multi_agent_analyst if USE_MULTI_AGENT else stock_analyst_agent
        max_turns = MAX_TURNS * 2 if USE_MULTI_AGENT else MAX_TURNS

        if USE_MULTI_AGENT:
            print("[Multi-Agent Mode]: Delegating to specialist analysts...\n")

        try:
            result = await Runner.run(
                agent,
                query,
                max_turns=max_turns,
            )
            print(f"\n{result.final_output}")

            # Check if PDF was generated
            for item in result.new_items:
                if hasattr(item, 'output') and 'pdf_path' in str(item.output):
                    print("\n[PDF Report has been generated!]")

        except Exception as e:
            print(f"\n[ERROR]: {str(e)}")
            print("\nPlease check:")
            print("1. Your LLM_PROVIDER and provider API key are set correctly")
            print("2. You have internet connectivity")
            print("3. The stock symbol is valid")


async def pipeline_interactive_session():
    """Run interactive CLI queries through the sequential pipeline."""
    print("\n" + "="*60)
    print("  INDIAN STOCK ANALYST AI (Pipeline System)")
    print("  Powered by OpenAI Agents SDK")
    print("="*60)
    print("\nWelcome! I can help you analyze Indian stocks (NSE/BSE).")
    print("Ask me about any stock, e.g., 'Should I buy RELIANCE?'")
    print("\n[Pipeline Mode Active]")
    print("- Runs specialist agents sequentially")
    print("- Avoids provider-specific handoff tool schema issues")
    print("- Generates a PDF report when analysis completes")
    print("\nType 'quit' or 'exit' to end the session.\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Indian Stock Analyst AI. Goodbye!")
                break

            print("\nAnalyzing... (this may take a moment)\n")
            stock_symbol = extract_stock_symbol(user_input)
            print(f"[Detected Stock]: {stock_symbol}")

            result = await run_full_analysis_with_pdf(stock_symbol, user_input)
            try:
                print(f"\n{result}")
            except UnicodeEncodeError:
                safe_result = result.encode('ascii', 'replace').decode('ascii')
                print(f"\n{safe_result}")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("\nPlease check:")
            print("1. Your LLM_PROVIDER and provider API key are set correctly")
            print("2. Your selected MODEL_NAME supports tool/function calling")
            print("3. The stock symbol is valid")


async def main():
    """Main entry point."""
    print_banner()

    # Check selected LLM provider configuration.
    if not is_llm_configured():
        print("\n[WARNING]: LLM provider is not fully configured!")
        print(f"Details: {get_configuration_error()}")
        print("Set LLM_PROVIDER plus the matching API key in .env, then restart.\n")

    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg in ['--help', '-h', 'help']:
            print_help()
            return

        # Single query mode
        query = ' '.join(sys.argv[1:])
        await single_query_mode(query)
    else:
        # Interactive mode
        provider_status = get_provider_status()
        print(f"\n[Provider]: {provider_status.get('provider')} / {provider_status.get('model')}")
        mode_str = "Pipeline (9 agents sequential)" if USE_PIPELINE_MODE else ("Multi-Agent" if USE_MULTI_AGENT else "Single-Agent")
        print(f"[Mode]: {mode_str}")
        print(f"[Reports]: Will be saved to: reports/")
        print("\nEnter a stock query (e.g., 'Analyze RELIANCE') or 'quit' to exit.\n")
        if USE_PIPELINE_MODE:
            await pipeline_interactive_session()
        else:
            await interactive_session(use_multi_agent=USE_MULTI_AGENT)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        sys.exit(1)
