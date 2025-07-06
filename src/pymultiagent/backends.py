"""
PyMultiAgent - Backend Management System

This module provides the backend configuration and management system for the PyMultiAgent framework.
It supports multiple LLM providers with a unified interface.
"""

import os
import logging
from typing import Dict, Any, Optional, List

try:
    from agents import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI, AsyncAzureOpenAI
except ImportError:
    raise ImportError(
        "Required packages not found. Please install with 'pip install openai openai-agents'"
    )

# Get module logger
logger = logging.getLogger(__name__)

# --- Backend Configuration ---

BACKEND_CONFIGS = {
    "llama": {
        "client_class": AsyncOpenAI,
        "api_key": os.getenv("LLAMA_OPENAI_API_KEY", "dummy"),
        "endpoint": os.getenv("LLAMA_OPENAI_ENDPOINT", "http://localhost:8000/v1/"),
        "models": {"llama3.2": {}, "llama3.3": {}},
        "client_kwargs": lambda cfg, model: {
            "api_key": cfg["api_key"],
            "base_url": cfg["endpoint"],
        },
    },
    "azure": {
        "client_class": AsyncAzureOpenAI,
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "endpoint": os.getenv(
            "AZURE_OPENAI_ENDPOINT", "https://openaiapimanagementxcse.azure-api.net"
        ),
        "models": {
            "gpt-4o": {"deployment": "gpt-4o", "api_version": "2025-01-01-preview"},
            "gpt-4o-mini": {
                "deployment": "gpt-4o-mini",
                "api_version": "2025-01-01-preview",
            },
            "o3-mini": {"deployment": "o3-mini", "api_version": "2025-01-01-preview"},
            "o4-mini": {"deployment": "o4-mini", "api_version": "2025-01-01-preview"},
        },
        "client_kwargs": lambda cfg, model: {
            "api_key": cfg["api_key"],
            "api_version": cfg["models"][model]["api_version"],
            "azure_endpoint": cfg["endpoint"],
            "azure_deployment": cfg["models"][model]["deployment"],
            "default_headers": {
                "Ocp-Apim-Subscription-Key": cfg["api_key"],
                "api-key": cfg["api_key"],
            },
        },
    },
    "openai": {
        "client_class": AsyncOpenAI,
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "endpoint": os.getenv("OPENAI_API_ENDPOINT", "https://api.openai.com/v1/"),
        "models": {
            "gpt-3.5-turbo": {},
            "gpt-4": {},
            "gpt-4-turbo": {},
            "gpt-4o": {},
            "gpt-4o-mini": {},
            "o3-mini": {},
            "o4-mini": {},
        },
        "client_kwargs": lambda cfg, model: {"api_key": cfg["api_key"]},
    },
}


# --- Factory Functions ---

def create_openai_client(backend: str, model_name: str):
    """
    Initializes the client for a given backend and model.

    Args:
        backend (str): The backend identifier (azure, openai, llama)
        model_name (str): The name of the model to use

    Returns:
        Client: An initialized OpenAI client

    Raises:
        ValueError: If the backend or model is not supported
    """
    config = BACKEND_CONFIGS.get(backend)
    if not config:
        raise ValueError(f"Unsupported backend: {backend}")

    if model_name not in config["models"]:
        raise ValueError(f"Unsupported model '{model_name}' for backend '{backend}'")

    client_class = config["client_class"]
    client = client_class(**config["client_kwargs"](config, model_name))
    return client


def get_chat_model(backend: str = "azure", model_name: str = "o4-mini"):
    """
    Returns an OpenAIChatCompletionsModel using the selected backend and model.

    Args:
        backend (str): The backend to use (azure, openai, llama)
        model_name (str): The model name to use

    Returns:
        OpenAIChatCompletionsModel: A configured chat model

    Raises:
        ValueError: If the backend or model is not supported
    """
    client = create_openai_client(backend, model_name)
    return OpenAIChatCompletionsModel(openai_client=client, model=model_name)


# --- Backend Class (for OOP compatibility) ---

class Backend:
    """Base class for LLM backends (for OOP compatibility)."""

    def __init__(self, backend_type: str):
        self.backend_type = backend_type
        self.config = BACKEND_CONFIGS.get(backend_type, {})
        self.models = self.config.get("models", {})

    def get_chat_model(self, model_name: str):
        """Get a chat model for this backend."""
        return get_chat_model(self.backend_type, model_name)

    def get_models(self) -> List[str]:
        """Get all models registered for this backend."""
        return list(self.models.keys())


class Backends:
    """Registry for accessing LLM backends."""

    @staticmethod
    def get_backend_types():
        """Get all available backend types."""
        return list(BACKEND_CONFIGS.keys())

    @staticmethod
    def get_backend(backend_type: str) -> Backend:
        """Get a backend instance by type."""
        if backend_type not in BACKEND_CONFIGS:
            raise ValueError(f"Unsupported backend: {backend_type}")
        return Backend(backend_type)

    @staticmethod
    def get_models_for_backend(backend_type: str) -> List[str]:
        """Get all models available for a specific backend."""
        config = BACKEND_CONFIGS.get(backend_type)
        if not config:
            raise ValueError(f"Unsupported backend: {backend_type}")
        return list(config.get("models", {}).keys())

    @staticmethod
    def validate_config(backend: str, model_name: str) -> bool:
        """Validate that a backend and model configuration is valid."""
        if backend not in BACKEND_CONFIGS:
            return False

        config = BACKEND_CONFIGS[backend]
        if model_name not in config.get("models", {}):
            return False

        return True


# --- Backward Compatibility Functions ---

def create_client_for_backend(backend: str, model_name: str):
    """Create a client for a specific backend and model (compatibility function)."""
    return create_openai_client(backend, model_name)


def get_available_backends():
    """Get list of available backends (compatibility function)."""
    return Backends.get_backend_types()


def get_available_models(backend: str):
    """Get list of available models for a backend (compatibility function)."""
    return Backends.get_models_for_backend(backend)


def initialize_backend_types():
    """
    Initialize the Backends registry with specialized backend instances.
    This is a compatibility function for backward compatibility.
    """
    # No need to do anything as backends are now defined in BACKEND_CONFIGS
    logger.info("Backend types initialized from BACKEND_CONFIGS")


def configure_backends():
    """
    Configure backends with runtime values.
    This is a compatibility function for backward compatibility.
    """
    # No need to do anything as backends are already configured in BACKEND_CONFIGS
    logger.info("Backends configured from BACKEND_CONFIGS and environment variables")
