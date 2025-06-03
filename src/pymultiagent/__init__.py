"""Python Multi-Agent Framework

A framework for creating and interacting with multiple AI agents.
"""

__version__ = "0.1.0"

from .assistants import Assistant
from .chat import AbstractChat, CLIChat, TelegramChat

__all__ = ["Assistant", "AbstractChat", "CLIChat", "TelegramChat"]
