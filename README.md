# PyMultiAgent

A simple yet powerful multi-agent framework built on top of the OpenAI Agent SDK.

## Overview

PyMultiAgent provides a flexible architecture for building AI agent systems with specialized capabilities. The framework implements a triage-based approach where a primary agent intelligently routes requests to specialized agents based on the nature of the query.

Key features:
- Triage-based multi-agent architecture
- Specialized agents for different types of tasks
- Extensive tool library for real-world interactions
- Support for multiple LLM backends (OpenAI, Azure OpenAI, Llama)
- Interactive chat mode with streaming responses
- Built-in test suite

## Installation

### Quick Installation

You can install PyMultiAgent directly from the repository:

```bash
# Using pip
pip install git+https://github.com/tbrandenburg/pymultiagent.git

# Using uv
uv pip install git+https://github.com/tbrandenburg/pymultiagent.git
```

### Development Installation

For development or customization:

```bash
# Clone the repository
git clone https://github.com/tbrandenburg/pymultiagent.git
cd pymultiagent

# Install in development mode with pip
pip install -e .

# Or with uv
uv pip install -e .
```

### Running the CLI

After installation, you can run the interactive chat mode using:

```bash
# Run the CLI command (simplest method)
pymultiagent

# Run with specific options
pymultiagent --backend azure --model gpt-4o --max_turns 20
```

Alternatively, you can run it as a Python module:

```bash
python -m pymultiagent
```

If you're using uv:

```bash
uv run python -m pymultiagent
```

### Environment Configuration

PyMultiAgent requires several API keys to enable full functionality. Create a `.env` file in your current working directory with the following keys:

```
# Azure OpenAI (primary backend)
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint

# OpenAI (alternative backend)
OPENAI_API_KEY=your_openai_api_key

# Llama (alternative backend)
LLAMA_OPENAI_API_KEY=your_llama_api_key
LLAMA_OPENAI_ENDPOINT=your_llama_endpoint

# Web Search (for knowledge assistant)
SERPER_API_KEY=your_serper_api_key

# Telegram Interface
TELEGRAM_BOT_TOKEN=your_telegram_bot_token  # Required for telegram interface
TELEGRAM_ALLOWED_USERS=123456789,987654321  # Comma-separated list of allowed user IDs (optional)
```

The system will first look for a `.env` file in your current working directory, and then fall back to the package installation directory if none is found.

### Prerequisites

- Python 3.8 or higher
- OpenAI API key, Azure OpenAI API key, or local Llama model

### Manual Setup

If you prefer to set up the project manually:

1. Clone the repository:
```bash
git clone https://github.com/tbrandenburg/pymultiagent.git
cd pymultiagent
```

2. Set up a virtual environment:

```bash
# Using Python's built-in venv
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# OR
# .venv\Scripts\activate  # On Windows

# Install in development mode
pip install -e .
```

Or with uv:

```bash
# Install uv if needed
pip install uv

# Create and use a virtual environment
uv venv
source .venv/bin/activate  # On Linux/Mac
# OR
# .venv\Scripts\activate  # On Windows

# Install in development mode
uv pip install -e .
```

3. Set up environment variables:
Create a `.env` file in your current working directory with your API keys like described above.

## Usage

### Command Line Interface

There are several ways to run PyMultiAgent after installation:

#### Option 1: Using the CLI command (Recommended)

After installing the package, you can use the `pymultiagent` command:

```bash
# Run with default settings
pymultiagent

# With specific backend and model
pymultiagent --backend azure --model gpt-4o

# Run test cases before interactive mode
pymultiagent --tests

# Use Telegram interface (requires TELEGRAM_BOT_TOKEN in .env file)
pymultiagent --interface telegram
```

#### Option 2: Using Python Module

You can run the project as a Python module:

```bash
# Run with default settings
python -m pymultiagent

# With arguments
python -m pymultiagent --backend azure --model gpt-4o --max_turns 20

# Run test cases
python -m pymultiagent --tests

# Use Telegram interface (requires TELEGRAM_BOT_TOKEN in .env file)
python -m pymultiagent --interface telegram
```

#### Option 3: Using UV Run (No virtual environment activation needed)

If you use UV, you can run the package directly:

```bash
# Run with default settings
uv run python -m pymultiagent

# With arguments (note the double dash)
uv run python -m pymultiagent -- --backend azure --model gpt-4o

# Run test cases
uv run python -m pymultiagent -- --tests

# Use Telegram interface (requires TELEGRAM_BOT_TOKEN in .env file)
uv run python -m pymultiagent -- --interface telegram
```

**Note**: When passing arguments with `uv run`, you must use a double dash (`--`) to separate the package name from its arguments.


### Command Line Options

- `--backend`: Choose the backend provider (azure, openai, llama)
- `--model`: Specify the model to use
- `--max_turns`: Maximum number of turns for agent interaction
- `--tests`: Run test cases before interactive mode
- `--no-interactive`: Skip interactive mode
- `--interface`: Choose the interface to use (cli, telegram)

### Interactive Chat

#### CLI Interface

In CLI interactive mode, you can:
- Type natural language requests
- Type `help` to see available commands
- Type `clear` to clear chat history
- Type `exit` to end the conversation

#### Telegram Interface

To use the Telegram interface:

1. Install the required dependency:
   ```bash
   pip install -e .[telegram]
   # or
   pip install python-telegram-bot
   ```

2. Create a Telegram bot using [@BotFather](https://t.me/BotFather) and get a token

3. Configure your Telegram token in your `.env` file:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   ```

4. Optionally restrict access to specific users in your `.env` file:
   ```
   TELEGRAM_ALLOWED_USERS=123456789,987654321
   ```

In Telegram chat, you can:
- Send natural language messages to the bot
- Use the `/help` command to see available commands
- Use the `/clear` command to clear chat history
- Use the `/start` command to restart the bot

## Available Agents

The framework includes several specialized agents:

1. **Triage Agent**: Routes requests to the appropriate specialized agent
2. **Date Assistant**: Provides current date information
3. **Time Assistant**: Provides current time information
4. **Coder Assistant**: Helps with programming tasks, file operations, and shell commands
5. **Knowledge Assistant**: Provides information by searching Wikipedia and the web
6. **Writing Assistant**: Assists with drafting and editing text content
7. **Math Assistant**: Helps with mathematical calculations and explanations

## Available Tools

PyMultiAgent includes a variety of tools for agents to interact with the environment:

### File Operations
- `read_file`: Read entire file content
- `read_file_lines`: Read specific lines from a file
- `write_file`: Write content to a file
- `check_directory_exists`: Check if a directory exists
- `check_file_exists`: Check if a file exists
- `get_current_working_directory`: Get current working directory
- `get_directory_tree`: Get directory structure
- `grep_files`: Search for patterns in files

### Information Retrieval
- `search_wikipedia`: Search Wikipedia articles
- `fetch_wikipedia_content`: Get content from Wikipedia
- `search_web_serper`: Perform web search using Serper API
- `fetch_http_url_content`: Fetch content from HTTP URLs

### Date and Time
- `get_current_date`: Get current date
- `get_current_time`: Get current time
- `get_current_datetime_rfc`: Get current datetime in RFC format

### Miscellaneous
- `execute_shell_command`: Run shell commands
- `get_user_input`: Get input from user
- `svg_text_to_png`: Convert SVG text to PNG

## Extending the Framework

### Adding a New Agent

To create a new specialized agent:

```python
new_agent = create_custom_agent(
 name='New Agent Name',
 instructions='Detailed instructions for the agent...',
 backend=backend,
 model_name=model_name,
 tools=[tool1, tool2, ...] # Optional tools for this agent
)
```

Then add it to the triage agent's handoffs:

```python
triage_agent = create_custom_agent(
 name='Triage Agent',
 instructions='...',
 backend=backend,
 model_name=model_name,
 handoffs=[..., new_agent] # Add your new agent here
)
```

### Adding a New Tool

Create a new function with the `@function_tool` decorator:

```python
@function_tool
def my_new_tool(param1: str, param2: int = 0) -> str:
 '''
 Tool description that will be shown to the agent.

 Args:
 param1: Description of param1
 param2: Description of param2

 Returns:
 Description of return value
 '''
 # Tool implementation
 return result
```

Then import and add it to the appropriate agent's tools list.

## Backend System

PyMultiAgent uses a modular backend system that allows for using different LLM providers with a unified interface:

### Backend Architecture

The backend system consists of:

1. **Backend Class Hierarchy**:
   - `Backend` (abstract base class): Defines the common interface for all backends
   - `AzureBackend`: Implementation for Azure OpenAI services
   - `OpenAIBackend`: Implementation for OpenAI API
   - `LlamaBackend`: Implementation for local Llama models

2. **Backends Registry**:
   - Central registry for all backends
   - Manages backend registration and configuration
   - Provides factory methods for creating clients and models

3. **Configuration Flow**:
   - At compile time: Backend types are registered with their client classes
   - At runtime: Backends are configured with API keys and endpoints
   - At runtime: Models are added to specific backends

### Using Different Backends

To use a specific backend and model:

```bash
# Using Azure backend with o4-mini model
uv run python -m pymultiagent -- --backend azure --model o4-mini

# Using OpenAI backend with gpt-4 model
uv run python -m pymultiagent -- --backend openai --model gpt-4

# Using local Llama backend
uv run python -m pymultiagent -- --backend llama --model llama3.3
```

### Extending with New Backends

To add a new backend provider:

1. Create a new backend class that inherits from the `Backend` base class
2. Implement the required methods, especially `create_client()`
3. Register the new backend during initialization

```python
class MyNewBackend(Backend):
    def __init__(self):
        super().__init__("my_new_backend")
        self.api_key = None
        self.endpoint = None

    def configure(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint
        return self

    def create_client(self, model_name):
        # Implementation for creating client
        ...

# Register the new backend
Backends.register(MyNewBackend())
```

## Architecture

PyMultiAgent follows a hierarchical architecture:

```
User Request <-> Triage Agent -> (User other assitant) ---+
                      ^                                   |
                      |                                   |
                      ------------------------------------+
```
