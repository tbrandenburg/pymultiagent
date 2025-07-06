"""
PyMultiAgent - Agent Management System

This module defines a system with a triage agent and specialized agents for
handling different types of requests.
"""

import os
from pathlib import Path
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union

from agents import Agent as OpenAIAgent, Runner
from pymultiagent.backends import get_chat_model
from pymultiagent.tools import function_tools

# Get module logger
logger = logging.getLogger(__name__)

class Agent:
    """
    A class representing an agent with configurable parameters.
    This wraps around the OpenAI Agent class and provides additional functionality.
    """

    def __init__(self, name, instructions, backend="azure", model_name="o4-mini", tools=None, handoffs=None):
        """
        Initialize an agent with the given parameters.
        The internal agent is created automatically during initialization.

        Args:
            name (str): The name of the agent
            instructions (str): The instructions for the agent, or a file path to a text file containing the prompt
            backend (str): The backend to use (azure, llama, openai)
            model_name (str): The model name to use for this agent
            tools (list, optional): List of tools available to the agent
            handoffs (list, optional): List of agents for handoffs (used by triage)
        """
        logger.info(f"Initializing agent: {name}")
        logger.debug(f"Agent parameters: backend={backend}, model={model_name}, tools={len(tools or [])}, handoffs={len(handoffs or [])}")

        self.name = name
        self.instructions = self._load_instructions(instructions)
        self.backend = backend
        self.model_name = model_name
        self.tools = tools or []
        self.handoffs = handoffs or []

        # Flag to track if we're in the process of creating an agent (for detecting circular dependencies)
        self._creating_agent = False

        # Create the agent automatically
        self._agent = self._create_agent()
        logger.info(f"Agent '{name}' initialized successfully")

    def _load_instructions(self, instructions):
        """
        Load instructions from a file if a file path is provided.

        Args:
            instructions (str): The instructions or a file path

        Returns:
            str: The loaded instructions
        """
        if isinstance(instructions, str) and os.path.isfile(instructions):
            logger.debug(f"Loading instructions from file: {instructions}")
            with open(instructions, "r", encoding="utf-8") as f:
                content = f.read()
                logger.debug(f"Loaded {len(content)} characters of instructions")
                return content
        elif isinstance(instructions, str) and "prompts/" in instructions:
            # Check for relative path in prompts directory
            prompts_directory = Path(__file__).parent / "prompts"
            sub_path = Path(instructions).relative_to("prompts/") if instructions.startswith("prompts/") else Path(instructions)
            full_path = prompts_directory / sub_path
            if full_path.is_file():
                logger.debug(f"Loading instructions from prompt file: {full_path}")
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    logger.debug(f"Loaded {len(content)} characters of instructions")
                    return content
        return instructions

    def get_agent(self):
        """
        Get the OpenAI agent for this agent.

        Returns:
            OpenAIAgent: The OpenAI agent for this agent
        """
        return self._agent

    def _get_handoff_agents(self):
        """
        Convert any Agent instances in handoffs to their OpenAI agents.
        Raises an exception if a circular dependency is detected.

        Returns:
            list: List of OpenAI agents for handoffs
        """
        logger.debug(f"Resolving handoff agents for agent '{self.name}'")
        handoff_agents = []
        for handoff in self.handoffs:
            if isinstance(handoff, Agent) and not isinstance(handoff, OpenAIAgent):
                # Check if we have a circular dependency
                if handoff._creating_agent:
                    logger.error(f"Circular dependency detected: {self.name} -> {handoff.name}")
                    raise ValueError(f"Circular dependency detected: {self.name} -> {handoff.name}")

                # Get the OpenAI agent from the handoff agent
                logger.debug(f"Getting agent from handoff agent: {handoff.name}")
                handoff_agents.append(handoff.get_agent())
            else:
                # Handle mock objects with _creating_agent attribute for tests
                if hasattr(handoff, '_creating_agent') and handoff._creating_agent:
                    logger.error(f"Circular dependency detected: {self.name} -> {handoff.name}")
                    raise ValueError(f"Circular dependency detected: {self.name} -> {handoff.name}")

                # Assume this is already an OpenAI Agent
                logger.debug(f"Using existing agent for handoff: {getattr(handoff, 'name', 'unnamed')}")
                handoff_agents.append(handoff)

        logger.debug(f"Resolved {len(handoff_agents)} handoff agents for agent '{self.name}'")
        return handoff_agents

    def _create_agent(self):
        """
        Private method to create the internal OpenAI Agent instance.

        Returns:
            OpenAIAgent: The created OpenAI Agent
        """
        logger.info(f"Creating backend agent for '{self.name}'")

        if self._creating_agent:
            logger.error(f"Circular dependency detected when creating {self.name}")
            raise ValueError(f"Circular dependency detected when creating {self.name}")

        try:
            # Set the flag to indicate we're creating an agent
            self._creating_agent = True
            logger.debug(f"Set creating_agent flag for '{self.name}'")

            # Get the chat model from the configured backend
            try:
                logger.debug(f"Getting chat model for backend '{self.backend}' and model '{self.model_name}'")
                model = get_chat_model(backend=self.backend, model_name=self.model_name)

                # Create the agent with all configured parameters
                kwargs = {
                    "name": self.name,
                    "instructions": self.instructions,
                    "model": model
                }

                if self.tools:
                    logger.debug(f"Adding {len(self.tools)} tools to agent '{self.name}'")
                    kwargs["tools"] = self.tools

                # Convert agent handoffs to OpenAI agents
                if self.handoffs:
                    logger.debug(f"Adding {len(self.handoffs)} handoffs to agent '{self.name}'")
                    kwargs["handoffs"] = self._get_handoff_agents()

                # Create and return the agent
                logger.info(f"Creating OpenAI Agent instance for '{self.name}'")
                return OpenAIAgent(**kwargs)
            except ValueError as e:
                # This allows tests to mock Agent creation while still being able
                # to test circular dependency detection
                if 'Backend' in str(e) and 'not registered' in str(e):
                    logger.warning(f"Backend not registered, creating mock agent for '{self.name}': {str(e)}")
                    return OpenAIAgent(name=self.name, instructions=self.instructions, model=None)
                logger.error(f"Error creating agent for '{self.name}': {str(e)}")
                raise
        finally:
            # Reset the flag
            self._creating_agent = False
            logger.debug(f"Reset creating_agent flag for '{self.name}'")

    def run_streamed(self, input_messages, max_turns=None):
        """
        Run the agent with streaming responses.

        Args:
            input_messages (list): The input messages or chat history
            max_turns (int, optional): Maximum number of turns for the conversation

        Returns:
            RunStream: A stream of responses from the agent
        """
        logger.info(f"Running agent '{self.name}' with streaming responses")
        logger.debug(f"Input message count: {len(input_messages)}, max_turns: {max_turns}")

        # Run the agent with streaming
        kwargs = {"input": input_messages}
        if max_turns is not None:
            kwargs["max_turns"] = max_turns

        # Try to get a running event loop, create one if none exists
        # This ensures the method works in both test and production environments
        try:
            asyncio.get_running_loop()
            logger.debug("Using existing event loop")
        except RuntimeError:
            # No running event loop, create a new one
            logger.debug("No running event loop, creating a new one")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        logger.debug(f"Starting streamed run for agent '{self.name}'")
        return Runner.run_streamed(self._agent, **kwargs)

    def __getattr__(self, name):
        """
        Delegate attribute access to the internal agent.
        This allows the Agent to be used like an OpenAI Agent.

        Args:
            name (str): The attribute name to access

        Returns:
            Any: The attribute value from the internal agent

        Raises:
            AttributeError: If the attribute does not exist
        """
        if hasattr(self._agent, name):
            return getattr(self._agent, name)

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

class AgentManager:
    """
    Manages the configuration of agents in the system.
    """

    def __init__(self, backend="azure", model_name="gpt-4o-mini"):
        """
        Initialize the AgentManager with the specified backend and model.

        Args:
            backend (str): The backend to use (azure, openai, llama)
            model_name (str): The model name to use for agents
        """
        self.backend = backend
        self.model_name = model_name
        self._validate_config()
        self.agents = {}
        logger.info(f"Initialized AgentManager with backend '{backend}' and model '{model_name}'")

    def _validate_config(self):
        """
        Validate the backend and model configuration.
        If the model is not available for the backend, use the first available model.
        """
        from pymultiagent.backends import Backends
        if self.backend not in Backends.get_backend_types():
            raise ValueError(f"Unsupported backend: {self.backend}")

        available_models = Backends.get_models_for_backend(self.backend)
        if self.model_name not in available_models:
            # Use the first available model for the backend if specified model is not available
            original_model = self.model_name
            self.model_name = available_models[0]
            logger.warning(f"Model '{original_model}' not available for backend '{self.backend}'. Using '{self.model_name}' instead.")

    def add_agent(self, agent_key, agent):
        """
        Add a specialized agent to the manager.

        Args:
            agent_key (str): The key to identify this agent (e.g., 'date', 'code')
            agent (Agent): The agent instance to add

        Returns:
            Agent: The added agent
        """
        self.agents[agent_key] = agent
        logger.info(f"Added agent '{agent.name}' with key '{agent_key}'")
        return agent

    def get_agents(self):
        """Get all registered agents.

        Returns:
            list: List of all agent instances registered with this manager.
        """
        return list(self.agents.values())

    def get_agent_info(self):
        """Get information about the current configuration."""
        return {
            "backend": self.backend,
            "model": self.model_name,
            "available_agents": list(self.agents.keys()) if self.agents else []
        }

    def create_custom_agent(self, agent_key, name, instructions, tools=None, handoffs=None):
        """
        Create a custom agent and add it to the manager.

        This method creates a new agent with the specified parameters,
        adds it to the manager with the given key, and returns both
        the agent object and its underlying OpenAI agent.

        Args:
            agent_key (str): The key to identify this agent in the manager
            name (str): The name of the agent
            instructions (str): The instructions for the agent, or a file path to a text file
            tools (list, optional): List of tools available to the agent
            handoffs (list, optional): List of agents for handoffs (used by triage)

        Returns:
            tuple: (Agent, OpenAIAgent) - The created agent and its underlying OpenAI agent
        """
        logger.info(f"Creating custom agent '{name}' with key '{agent_key}'")
        agent = Agent(
            name=name,
            instructions=instructions,
            backend=self.backend,
            model_name=self.model_name,
            tools=tools,
            handoffs=handoffs
        )

        # Add the agent to the manager
        self.add_agent(agent_key, agent)

        # Return both the agent and its underlying OpenAI agent
        openai_agent = agent.get_agent()
        logger.info(f"Created custom agent '{name}' with key '{agent_key}'")
        return agent, openai_agent



# Function moved to chat.py
