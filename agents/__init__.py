"""
Specialized agents for multi-agent stock analysis system.
"""

# Import SDK components from the wrapper module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai_sdk import Agent, Runner, ModelSettings, function_tool, SDK_AVAILABLE, handoff

# Now import local agent modules
from .fundamental_analyst import fundamental_analyst_agent
from .technical_analyst import technical_analyst_agent
from .sentiment_analyst import sentiment_analyst_agent
from .macro_analyst import macro_analyst_agent
from .document_analyst import document_analyst_agent
from .bull_agent import bull_agent
from .bear_agent import bear_agent
from .debate_judge import debate_judge_agent
from .risk_manager import risk_manager_agent
from .portfolio_analyst import portfolio_analyst_agent
from .orchestrator import stock_orchestrator_agent

# Do not add reverse handoffs from specialists back to the orchestrator here.
# The sequential pipeline invokes each specialist directly, and OpenAI-compatible
# BYOLLM providers such as Groq can reject SDK-generated handoff tool schemas.

__all__ = [
    "Agent", "Runner", "ModelSettings", "function_tool", "SDK_AVAILABLE", "handoff",
    "fundamental_analyst_agent", "technical_analyst_agent", "sentiment_analyst_agent",
    "macro_analyst_agent", "document_analyst_agent",
    "bull_agent", "bear_agent", "debate_judge_agent",
    "risk_manager_agent", "portfolio_analyst_agent",
    "stock_orchestrator_agent",
]
