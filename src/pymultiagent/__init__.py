"""Python Multi-Agent Framework

A framework for creating and interacting with multiple AI agents.
"""

__version__ = "0.1.0"

from .agents import Agent, AgentManager
from .main import create_specialized_agents, create_triage_agent
from .chat import AbstractChat, CLIChat, TelegramChat, process_chat_history

__all__ = ["Agent", "AgentManager", "create_specialized_agents", "create_triage_agent", "AbstractChat", "CLIChat", "TelegramChat", "process_chat_history"]
