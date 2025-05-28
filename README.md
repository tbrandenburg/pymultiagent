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

### Prerequisites

- Python 3.8 or higher
- OpenAI API key, Azure OpenAI API key, or local Llama model

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pymultiagent.git
cd pymultiagent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the project root with your API keys:

```
# For OpenAI backend
OPENAI_API_KEY=your_openai_api_key

# For Azure OpenAI backend
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint

# For Llama local model
LLAMA_OPENAI_API_KEY=dummy
LLAMA_OPENAI_ENDPOINT=http://localhost:8000/v1/

# Optional: Serper API for web search
SERPER_API_KEY=your_serper_api_key
```

## Usage

### Command Line Interface

Run the main application with default settings:
```bash
python main.py
```

Run with specific backend and model:
```bash
python main.py --backend azure --model gpt-4o
```

Run test cases:
```bash
python main.py --tests
```

### Command Line Options

- `--backend`: Choose the backend provider (azure, openai, llama)
- `--model`: Specify the model to use
- `--max_turns`: Maximum number of turns for agent interaction
- `--tests`: Run test cases before interactive mode
- `--no-interactive`: Skip interactive mode

### Interactive Chat

In interactive mode, you can:
- Type natural language requests
- Type `help` to see available commands
- Type `clear` to clear chat history
- Type `exit` to end the conversation

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

## Architecture

PyMultiAgent follows a hierarchical architecture:

```
User Request <-> Triage Agent -> (User other assitant) ---+
                      ^                                   |
                      |                                   |
                      ------------------------------------+
```
