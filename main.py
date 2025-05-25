"""
PyMultiAgent - A multi-agent system for handling different types of requests.

This module defines a system with a triage agent and specialized agents for
handling date, time, and coding requests.
"""
import os
import asyncio
import argparse
from dotenv import load_dotenv

try:
    from agents import Agent, OpenAIChatCompletionsModel, Runner
    from openai import AsyncOpenAI, AsyncAzureOpenAI
except ImportError:
    raise ImportError("Required packages not found. Please install with 'pip install openai openai-agents'")

# Import tools and tests from separate modules
try:
    from .tools import get_current_date, get_current_time, read_file, read_file_lines, write_file, execute_shell_command, search_wikipedia, fetch_wikipedia_content, search_web_serper, fetch_http_url_content
    from .tests import run_test_cases, format_separator_line
except ImportError:
    # If relative imports fail, try absolute imports for direct execution
    from tools import get_current_date, get_current_time, read_file, read_file_lines, write_file, execute_shell_command, search_wikipedia, fetch_wikipedia_content, search_web_serper, fetch_http_url_content
    from tests import run_test_cases, format_separator_line

# Load environment variables
load_dotenv()

# --- Backend Configuration ---

BACKEND_CONFIGS = {
    "llama": {
        "client_class": AsyncOpenAI,
        "api_key": os.getenv("LLAMA_OPENAI_API_KEY", "dummy"),
        "endpoint": os.getenv("LLAMA_OPENAI_ENDPOINT", "http://localhost:8000/v1/"),
        "models": {
            "llama3.2": {},
            "llama3.3": {}
        },
        "client_kwargs": lambda cfg, model: {
            "api_key": cfg["api_key"],
            "base_url": cfg["endpoint"]
        }
    },
    "azure": {
        "client_class": AsyncAzureOpenAI,
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "https://openaiapimanagementxcse.azure-api.net"),
        "models": {
            "gpt-4o": {
                "deployment": "gpt-4o",
                "api_version": "2025-01-01-preview"
            },
            "gpt-4o-mini": {
                "deployment": "gpt-4o-mini",
                "api_version": "2025-01-01-preview"
            },
            "o3-mini": {
                "deployment": "o3-mini",
                "api_version": "2025-01-01-preview"
            },
            "o4-mini": {
                "deployment": "o4-mini",
                "api_version": "2025-01-01-preview"
            }
        },
        "client_kwargs": lambda cfg, model: {
            "api_key": cfg["api_key"],
            "api_version": cfg["models"][model]["api_version"],
            "azure_endpoint": cfg["endpoint"],
            "azure_deployment": cfg["models"][model]["deployment"],
            "default_headers": {
                "Ocp-Apim-Subscription-Key": cfg["api_key"],
                "api-key": cfg["api_key"]
            }
        }
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
            "o4-mini": {}
        },
        "client_kwargs": lambda cfg, model: {
            "api_key": cfg["api_key"]
        }
    }
}

# --- Factory Functions ---

def create_openai_client(backend: str, model_name: str):
    """
    Initializes the client for a given backend and model.
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
    """
    client = create_openai_client(backend, model_name)
    return OpenAIChatCompletionsModel(openai_client=client, model=model_name)


# --- Agent Creation ---

def create_custom_agent(name, instructions, backend="azure", model_name="o4-mini", tools=None, handoffs=None):
    """
    Create a custom agent with specified parameters and internal model creation.

    Args:
        name (str): The name of the agent
        instructions (str): The instructions for the agent, or a file path to a text file containing the agent prompt
        backend (str): The backend to use (azure, llama)
        model_name (str): The model name to use for this agent
        tools (list, optional): List of tools available to the agent
        handoffs (list, optional): List of agents for handoffs (used by triage)

    Returns:
        Agent: Configured custom agent
    """
    import os
    if os.path.isfile(instructions):
        with open(instructions, "r") as f:
            instructions = f.read()
    model = get_chat_model(backend=backend, model_name=model_name)
    kwargs = {"name": name, "instructions": instructions, "model": model}
    if tools:
        kwargs["tools"] = tools
    if handoffs:
        kwargs["handoffs"] = handoffs
    return Agent(**kwargs)

# --- Utility Functions ---

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
  %(prog)s --backend azure --model gpt-4o
  %(prog)s --backend llama --model llama3.3 --tests
  %(prog)s --tests --no-interactive
        """
    )

    # Get available backends and models for help text
    available_backends = list(BACKEND_CONFIGS.keys())

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
        "--tests",
        action="store_true",
        help="Run test cases before interactive mode"
    )

    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive mode"
    )

    args = parser.parse_args()

    # Convert to dictionary format for backward compatibility
    config = {
        "backend": args.backend,
        "model": args.model,
        "interactive": not args.no_interactive,
        "run_tests": args.tests
    }

    return config

def process_chat_history(chat_history, new_response):
    """
    Update chat history with a new response.

    Args:
        chat_history (list): Existing chat history
        new_response: New response to add to history

    Returns:
        list: Updated chat history
    """
    return chat_history + new_response.to_input_list()



# --- Interactive Chat ---

async def run_interactive_chat(triage_agent):
    """
    Run an interactive chat session with the user, streaming responses.

    Args:
        triage_agent: The initialized triage agent
    """
    print("--- Interactive Chat Mode ---")
    print("Type 'exit' to end the conversation")
    print("Type 'help' for available commands")

    chat_history = []

    while True:
        prompt = input("\nEnter the next prompt (or 'exit'): ")

        # Handle special commands
        if prompt.lower() == "exit":
            print("Exiting chat mode. Goodbye!")
            break
        elif prompt.lower() == "help":
            print("\nAvailable commands:")
            print("  exit - End the conversation")
            print("  help - Show this help message")
            print("  clear - Clear the chat history")
            continue
        elif prompt.lower() == "clear":
            chat_history = []
            print("Chat history cleared.")
            continue

        # Add user input to chat history
        chat_history = chat_history + [{"role": "user", "content": prompt}]

        print("Processing your request (streaming)...")

        try:
            # Stream the response from the agent system
            streamed_output = ""
            last_agent = None
            stream = Runner.run_streamed(
                triage_agent,
                input=chat_history,
            )
            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    from openai.types.responses import ResponseTextDeltaEvent
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        print(event.data.delta, end="", flush=True)
                        streamed_output += event.data.delta
                elif event.type == "agent_updated_stream_event":
                    last_agent = event.new_agent
                    print(f"\nAgent updated: {last_agent.name}", flush=True)
                elif event.type == "run_item_stream_event":
                    from agents import ItemHelpers
                    if event.item.type == "tool_call_item":
                        print("\n-- Tool was called", flush=True)
                    elif event.item.type == "tool_call_output_item":
                        print(f"\n-- Tool output: {event.item.output}", flush=True)
                    elif event.item.type == "message_output_item":
                        text = ItemHelpers.text_message_output(event.item)
                        print(f"\n-- Message output:\n{text}", flush=True)
                        # If no token output has been printed, use this complete text as final output.
                        if not streamed_output:
                            streamed_output = text

            print()  # Newline after streaming output

            if streamed_output:
                print(f"\nFinal response: {streamed_output}")
            if last_agent:
                print(f"Agent responsible: {last_agent.name}")

            # Update chat history with the streamed response
            chat_history = chat_history + [{"role": "assistant", "content": streamed_output}]
        except Exception as e:
            print(f"\nError processing request: {e}")
            print("Your message was not added to the chat history.")
            if chat_history:
                chat_history.pop()

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

    try:
        # Validate backend and model configuration
        backend = args["backend"]
        model_name = args["model"]

        if backend not in BACKEND_CONFIGS:
            raise ValueError(f"Unsupported backend: {backend}")

        if model_name not in BACKEND_CONFIGS[backend]["models"]:
            # Use the first available model for the backend if specified model is not available
            available_models = list(BACKEND_CONFIGS[backend]["models"].keys())
            model_name = available_models[0]
            print(f"Model '{args['model']}' not available for backend '{backend}'. Using '{model_name}' instead.")

        # Create agent models with different configuration settings via create_custom_agent
        date_assistant = create_custom_agent(
            name="Date Assistant",
            instructions=(
                "You are a helpful date assistant. Be concise and professional. "
                "You must use the 'get_current_date' tool to obtain the date. Do not attempt to provide the date without using the tool."
            ),
            backend=backend,
            model_name=model_name,
            tools=[get_current_date]
        )

        time_assistant = create_custom_agent(
            name="Time Assistant",
            instructions=(
                "You are a helpful time assistant. Be concise and professional. "
                "You must use the 'get_current_time' tool to obtain the time. Do not attempt to provide the time without using the tool."
            ),
            backend=backend,
            model_name=model_name,
            tools=[get_current_time]
        )

        code_assistant = create_custom_agent(
            name="Coder Assistant",
            instructions="prompts/coder_prompt.txt",
            backend=backend,
            model_name=model_name,
            tools=[read_file, read_file_lines, write_file, execute_shell_command]
        )

        knowledge_assistant = create_custom_agent(
            name="Knowledge Assistant",
            instructions=(
                "You are a helpful knowledge assistant. You are highly knowledgeable across a wide range of topics and disciplines."
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
            tools=[search_wikipedia, fetch_wikipedia_content, search_web_serper, fetch_http_url_content]
        )

        writing_assistant = create_custom_agent(
            name="Writing Assistant",
            instructions=(
                "You are a professional writing assistant. You excel at creating clear, engaging, and error-free text."
                "You can help with essays, articles, blog posts, emails, and other written content."
                "You provide thoughtful suggestions for improving clarity, flow, and style."
            ),
            backend=backend,
            model_name=model_name,
            tools=[]  # No specialized tools for this example
        )

        math_assistant = create_custom_agent(
            name="Math Assistant",
            instructions=(
                "You are a specialized mathematics assistant. You can help with calculations, "
                "mathematical proofs, and explaining mathematical concepts in an accessible way. "
                "Always show your work step by step when solving problems."
            ),
            backend=backend,
            model_name=model_name,
            tools=[]  # No specialized tools for this example
        )

        # Create triage agent via create_custom_agent with handoffs
        triage_agent = create_custom_agent(
            name="Triage Agent",
            instructions=(
                "You are a general-purpose Triage Agent. Your primary role is to understand the user's intent "
                "and route their request to the most appropriate specialized assistant. "
                "Be concise and professional. Do not answer questions that specialized assistants can handle, always hand off."
            ),
            backend=backend,
            model_name=model_name,
            handoffs = [date_assistant, time_assistant, code_assistant, writing_assistant, math_assistant, knowledge_assistant]
        )

        # Print summary of available agents
        print(f"\n--- Agent System initialized with backend '{backend}' and model '{model_name}' ---")
        print("\n" + format_separator_line() + "\n")

        # Run test cases if requested
        if args["run_tests"]:
            await run_test_cases(triage_agent)

        # Run interactive chat if requested
        if args["interactive"]:
            await run_interactive_chat(triage_agent)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    asyncio.run(main())
