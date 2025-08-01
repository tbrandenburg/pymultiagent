#!/usr/bin/env python
"""
PyMultiAgent - A multi-agent system for handling different types of requests.

This module defines a system with a triage agent and specialized agents for
handling various types of requests such as date, time, coding, knowledge,
writing, and math tasks.
"""
import os
import asyncio
import argparse
import logging
from dotenv import load_dotenv

from pymultiagent.backends import initialize_backend_types, configure_backends, Backends
from pymultiagent.agents import AgentManager, Agent
from pymultiagent.chat import CLIChat, TelegramChat, process_chat_history
from pymultiagent.tools import get_current_date, get_current_time, read_file, read_file_lines, write_file, execute_shell_command, search_wikipedia, fetch_wikipedia_content, search_web_serper, fetch_http_url_content, check_directory_exists, check_file_exists, get_current_working_directory, get_directory_tree, grep_files, svg_text_to_png
from pymultiagent.tests import run_test_cases, format_separator_line

# Load environment variables
# First check for .env in current working directory
cwd_env_path = os.path.join(os.getcwd(), '.env')
package_env_path = os.path.join(os.path.dirname(__file__), '.env')

if os.path.exists(cwd_env_path):
    load_dotenv(cwd_env_path)
    print(f"Loaded environment from {cwd_env_path}")
elif os.path.exists(package_env_path):
    load_dotenv(package_env_path)
    print(f"Loaded environment from {package_env_path}")
else:
    print("Warning: No .env file found in current directory or package directory")
    print("The following API keys may be required for full functionality:")
    print("  - AZURE_OPENAI_API_KEY: For Azure OpenAI models")
    print("  - OPENAI_API_KEY: For OpenAI models")
    print("  - LLAMA_OPENAI_API_KEY: For Llama models")
    print("  - SERPER_API_KEY: For web search functionality")
    print("  - TELEGRAM_BOT_TOKEN: For Telegram interface")
    print("  - TELEGRAM_ALLOWED_USERS: Comma-separated list of allowed Telegram user IDs (optional)")
    print("Create a .env file with these keys to enable all features.")

# Configure logging
# File handler gets everything including tracebacks
file_handler = logging.FileHandler("pymultiagent.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Console handler gets clean messages without tracebacks
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, file_handler]
)

# Set HTTP-related and external library loggers to WARNING level to reduce API call logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize and configure backends (compatibility functions)
initialize_backend_types()
logger.info("Backend types initialized")

# Configure backends with API keys, endpoints, and models
configure_backends()
logger.info("Backends configured")

# --- Utility Functions ---

def format_user_friendly_error(error):
    """
    Format an error message in a user-friendly way.

    Args:
        error: The exception object

    Returns:
        str: A formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)

    # Handle known error types with custom messages
    if "Azure backend not fully configured" in error_msg:
        return (
            "Configuration Error: Azure OpenAI credentials are missing.\n\n"
            "Please set the following environment variables in your .env file:\n"
            "  AZURE_OPENAI_API_KEY=your_api_key\n"
            "  AZURE_OPENAI_ENDPOINT=your_endpoint_url\n\n"
            "You can also use '--backend openai' if you have OpenAI credentials configured."
        )
    elif "OpenAI backend not fully configured" in error_msg:
        return (
            "Configuration Error: OpenAI credentials are missing.\n\n"
            "Please set the following environment variable in your .env file:\n"
            "  OPENAI_API_KEY=your_api_key\n\n"
            "You can also use '--backend llama' if you have a local Llama model running with OpenAI-compatible API."
        )
    elif "Llama backend not fully configured" in error_msg:
        return (
            "Configuration Error: Llama model API credentials are missing.\n\n"
            "Please set the following environment variables in your .env file:\n"
            "  LLAMA_OPENAI_API_KEY=your_api_key\n"
            "  LLAMA_OPENAI_ENDPOINT=your_endpoint_url\n\n"
            "You can also use '--backend azure' or '--backend openai' if you have those credentials configured."
        )
    elif "Model" in error_msg and "not registered" in error_msg:
        return (
            f"Configuration Error: {error_msg}\n\n"
            "Please check that you specified a valid model for the selected backend.\n"
            "Use --help to see available models for each backend."
        )
    elif "Unsupported backend" in error_msg:
        return (
            f"Configuration Error: {error_msg}\n\n"
            "Please check that you specified a valid backend.\n"
            "Use --help to see available backends."
        )
    elif "Backend" in error_msg and "not registered" in error_msg:
        return (
            f"Configuration Error: {error_msg}\n\n"
            "Please check that you specified a valid backend.\n"
            "Use --help to see available backends."
        )
    elif "TELEGRAM_BOT_TOKEN" in error_msg:
        return (
            "Configuration Error: Telegram bot token is missing.\n\n"
            "Please set the following environment variable in your .env file:\n"
            "  TELEGRAM_BOT_TOKEN=your_bot_token\n\n"
            "Or pass it as a command line argument with --telegram-token"
        )
    elif "You cannot use the Telegram interface" in error_msg:
        return (
            "Configuration Error: Telegram interface cannot be used.\n\n"
            + error_msg
        )
    elif "KeyboardInterrupt" in error_type:
        return "Operation cancelled by user."

    # Default message for unknown errors
    return f"Error: {error_msg}\n\nType: {error_type}"


def parse_command_line_args():
    """
    Parse command line arguments using argparse.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="PyMultiAgent - A multi-agent system for handling different types of requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available backends and models:
  azure: gpt-4o, gpt-4o-mini, o1, o1-mini
  llama: llama3.3
  openai: gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini

Examples:
  %(prog)s                                    # Use default settings
  %(prog)s --backend azure --model gpt-4o --max_turns 20
  %(prog)s --backend llama --model llama3.3 --tests
  %(prog)s --tests --no-interactive
  %(prog)s --interface telegram --telegram-token YOUR_TOKEN
        """
    )

    # Get available backends and models for help text
    available_backends = Backends.get_backend_types()

    parser.add_argument(
        "--backend",
        choices=available_backends,
        default="azure",
        help="Backend to use for AI models (default: %(default)s)"
    )

    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use (default: %(default)s). Available models depend on backend."
    )

    parser.add_argument(
        "--max_turns",
        type=int,
        default=15,
        help="Maximum number of turns for agent interaction (default: %(default)s)"
    )

    parser.add_argument(
        "--tests",
        action="store_true",
        help="Run test cases before interactive mode"
    )

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive mode"
    )

    parser.add_argument(
        "--interface",
        choices=["cli", "telegram"],
        default="cli",
        help="Interface to use for interaction (default: %(default)s)"
    )

    args = parser.parse_args()

    # Get Telegram token from .env file
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")

    # Parse the telegram allowed users from environment variable
    telegram_allowed_users = None
    if os.getenv("TELEGRAM_ALLOWED_USERS") is not None:
        # Parse from environment variable
        telegram_allowed_users_env = os.getenv("TELEGRAM_ALLOWED_USERS")
        if telegram_allowed_users_env:  # Check if not empty string
            try:
                telegram_allowed_users = [int(user_id.strip()) for user_id in telegram_allowed_users_env.split(",") if user_id.strip()]
            except ValueError as e:
                print(f"Warning: Invalid Telegram user ID in environment variable: {str(e)}. All IDs must be integers.")

    # Convert to dictionary format for backward compatibility
    config = {
        "backend": args.backend,
        "model": args.model,
        "max_turns": args.max_turns,
        "interactive": not args.no_interactive,
        "run_tests": args.tests,
        "interface": args.interface,
        "telegram_token": telegram_token,
        "telegram_allowed_users": telegram_allowed_users
    }

    # Validate backend and model configuration
    backend = config["backend"]
    model_name = config["model"]

    # Validate interface configuration
    if config["interface"] == "telegram" and not config["telegram_token"]:
        parser.error("Telegram token is required when using the telegram interface. Please set TELEGRAM_BOT_TOKEN in your .env file.")

    if backend not in Backends.get_backend_types():
        raise ValueError(f"Unsupported backend: {backend}")

    backend_instance = Backends.get_backend(backend)
    available_models = backend_instance.get_models()

    if model_name not in available_models and available_models:
        # Use the first available model for the backend if specified model is not available
        model_name = available_models[0]
        print(f"Model '{config['model']}' not available for backend '{backend}'. Using '{model_name}' instead.")
        config["model"] = model_name

    return config

# --- Agent Creation Functions ---

def create_specialized_agents(backend="azure", model_name="gpt-4o-mini"):
    """
    Create all specialized agents and return them in a dictionary.

    Args:
        backend (str): The backend to use (azure, openai, llama)
        model_name (str): The model name to use for the agents

    Returns:
        dict: Dictionary of specialized agents
    """
    from pymultiagent.tools.function_tools import (
        get_current_date, get_current_time,
        read_file, write_file, check_directory_exists, check_file_exists,
        get_current_working_directory, execute_shell_command, get_directory_tree,
        grep_files, search_wikipedia, fetch_wikipedia_content, search_web_serper,
        fetch_http_url_content,
    )

    logger.info("Creating specialized agents")
    agents = {}

    # Date Agent
    agents["date"] = Agent(
        name="Date Agent",
        instructions=(
            "You are a helpful date agent. Be concise and professional. "
            "You must use the 'get_current_date' tool to obtain the date. Do not attempt to provide the date without using the tool."
        ),
        backend=backend,
        model_name=model_name,
        tools=[get_current_date],
    )

    # Time Agent
    agents["time"] = Agent(
        name="Time Agent",
        instructions=(
            "You are a helpful time agent. Be concise and professional. "
            "You must use the 'get_current_time' tool to obtain the time. Do not attempt to provide the time without using the tool."
        ),
        backend=backend,
        model_name=model_name,
        tools=[get_current_time],
    )

    # Code Agent
    agents["code"] = Agent(
        name="Coder Agent",
        instructions="prompts/coder_prompt.txt",
        backend=backend,
        model_name=model_name,
        tools=[
            read_file,
            write_file,
            check_directory_exists,
            check_file_exists,
            get_current_working_directory,
            execute_shell_command,
            get_directory_tree,
            grep_files,
        ],
    )

    # Knowledge Agent
    agents["knowledge"] = Agent(
        name="Knowledge Agent",
        instructions=(
            "You are a helpful knowledge agent. You are highly knowledgeable across a wide range of topics and disciplines."
            "You take care to provide information that is understandable, accurate, thorough, and reliable."
            "When providing common knowledge, you augment it with current information based on latest research and news."
            "You have access to tools to assist you."
            "When answering questions, follow these steps:"
            "1. **Get information:** Search for and gather comprehensive background information about the topic using Wikipedia and other reliable sources."
            "2. **Get latest news:** Use web search tools to find the most recent news or updates related to the topic."
            "3. **Summarize and explain:** Provide a clear, concise answer, and briefly explain your reasoning or summarize the key points to ensure clarity and understanding."
            "Use the available tools as needed to ensure your answers are accurate and up to date."
        ),
        backend=backend,
        model_name=model_name,
        tools=[
            search_wikipedia,
            fetch_wikipedia_content,
            search_web_serper,
            fetch_http_url_content,
        ],
    )

    # Writing Agent
    agents["writing"] = Agent(
        name="Writing Agent",
        instructions=(
            "You are a professional writing agent. You excel at creating clear, engaging, and error-free text."
            "You can help with essays, articles, blog posts, emails, and other written content."
            "You provide thoughtful suggestions for improving clarity, flow, and style."
        ),
        backend=backend,
        model_name=model_name,
        tools=[],
    )

    # Math Agent
    agents["math"] = Agent(
        name="Math Agent",
        instructions=(
            "You are a specialized mathematics agent. You can help with calculations, "
            "mathematical proofs, and explaining mathematical concepts in an accessible way. "
            "Always show your work step by step when solving problems."
        ),
        backend=backend,
        model_name=model_name,
        tools=[],
    )

    logger.info(f"Created {len(agents)} specialized agents")
    return agents

def create_triage_agent(backend="azure", model_name="gpt-4o-mini", specialized_agents=None):
    """
    Create a triage agent that can hand off to specialized agents.

    Args:
        backend (str): The backend to use (azure, openai, llama)
        model_name (str): The model name to use for the agent
        specialized_agents (dict): Dictionary of specialized agents for handoffs

    Returns:
        Assistant: The configured triage agent
    """
    if not specialized_agents:
        specialized_agents = {}
        logger.warning("No specialized agents provided to triage agent")

    logger.info("Creating triage agent directly")
    triage_agent = Agent(
        name="Triage Agent",
        instructions=(
            "You are a general-purpose Triage Agent. Your primary role is to understand the user's intent "
            "and route their request to the most appropriate specialized agent. "
            "Be concise and professional. Do not answer questions that specialized agents can handle, always hand off."
        ),
        backend=backend,
        model_name=model_name,
        handoffs=list(specialized_agents.values()),
    )
    logger.info("Triage agent created successfully")

    return triage_agent

# --- Main Function ---

async def main():
    """
    Main application entry point.

    This function initializes the client, creates agents, and runs the appropriate mode
    based on configuration settings.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Parse command line arguments
    args = parse_command_line_args()
    logger.info(f"Command line arguments: {args}")

    try:
        # Get backend and model configuration
        backend = args["backend"]
        model_name = args["model"]
        logger.info(f"Using backend: {backend}, model: {model_name}")

        # Create specialized agents
        specialized_agents = create_specialized_agents(backend=backend, model_name=model_name)

        # Create triage agent directly instead of through AgentManager
        triage_agent = create_triage_agent(
            backend=backend,
            model_name=model_name,
            specialized_agents=specialized_agents
        )

        # Initialize the AgentManager with command line arguments and add agents
        agent_manager = AgentManager(backend=backend, model_name=model_name)

        # Add all specialized agents to the manager
        for agent_key, agent in specialized_agents.items():
            agent_manager.add_agent(agent_key, agent)

        # Get agent information
        agent_info = {
            "backend": backend,
            "model": model_name,
            "available_agents": list(specialized_agents.keys()),
        }

        # Print summary of available agents
        print(f"\n--- Agent System initialized with backend '{agent_info['backend']}' and model '{agent_info['model']}' ---")
        print("\n" + format_separator_line() + "\n")
        logger.info(f"Agent system initialized with backend '{backend}' and model '{model_name}'")

        # Run test cases if requested
        # Run tests or interactive mode
        if args["run_tests"]:
            logger.info("Running test cases")
            await run_test_cases(triage_agent)
            logger.info("Test cases completed")

        # Run interactive chat if requested, passing max_turns
        if args["interactive"]:
            chat = None
            if args["interface"] == "cli":
                logger.info("Starting CLI interface")
                chat = CLIChat(triage_agent, max_turns=args["max_turns"])
            elif args["interface"] == "telegram":
                logger.info("Starting Telegram interface")
                verify_ssl = args.get("verify_ssl", True)
                if not verify_ssl:
                    logger.warning("SSL certificate verification is disabled. This is insecure!")
                chat = TelegramChat(
                    triage_agent,
                    token=args["telegram_token"],
                    max_turns=args["max_turns"],
                    allowed_user_ids=args["telegram_allowed_users"] or []
                )
            if chat:
                try:
                    await chat.run()
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt received, shutting down...")
                except Exception as e:
                    logger.error(f"Error in interface: {e}")
                finally:
                    logger.info(f"{args['interface'].upper()} interface terminated")

    except Exception as e:
        logger.exception(f"Error in main function: {e}")
        print(f"Error: {e}")
        return 1

    return 0

def cli_main():
    """
    Command-line interface entry point.
    This function is the entry point for the 'pymultiagent' command.
    """
    try:
        logger.info("Starting PyMultiAgent")
        # Handle keyboard interrupts at this level to avoid traceback output
        try:
            return asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down gracefully...")
            return 0
    except Exception as e:
        logger.exception(f"Unhandled exception in PyMultiAgent: {e}")
        return 1
    finally:
        logger.info("PyMultiAgent terminated")

if __name__ == "__main__":
    cli_main()
