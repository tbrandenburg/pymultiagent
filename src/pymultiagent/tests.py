"""
Test cases for the PyMultiAgent system.

This module contains predefined test cases for evaluating the triage agent
and specialized agents in the multi-agent system.
"""
import time

try:
    from agents import Runner
except ImportError:
    raise ImportError("Required packages not found. Please install with 'pip install openai openai-agents'")


def sum_total_tokens(raw_responses):
    """
    Calculate the total tokens used across all responses.

    Args:
        raw_responses: List of response objects with usage information

    Returns:
        int: Total number of tokens used
    """
    total_tokens = 0
    for response in raw_responses:
        if hasattr(response, 'usage') and response.usage:
            if hasattr(response.usage, 'total_tokens'):
                total_tokens += response.usage.total_tokens
    return total_tokens


def format_separator_line():
    """
    Format a separator line for output display.

    Returns:
        str: A formatted separator line
    """
    return "=" * 80


def print_agent_inner_dialog(result):
    """
    Print the agents' inner dialog.

    Each line contains the role and the first 100 characters of the message.
    For tool calls, use 'tool' as the role and the function name as content.

    Args:
        result: The result object from Runner.run() containing new_items
    """
    print("\nInner dialogue:")
    if hasattr(result, 'new_items') and result.new_items:
        for item in result.new_items:
            if hasattr(item, 'raw_item') and item.raw_item:
                if isinstance(item.raw_item, dict):
                    if 'type' in item.raw_item:
                        if 'output' in item.raw_item:
                            print(f"  {item.raw_item['type']}: {item.raw_item['output'][:100]}...")
                elif hasattr(item.raw_item, 'type'):
                    if hasattr(item.raw_item, 'name') and hasattr(item.raw_item, 'arguments'):
                        print(f"  {item.raw_item.type}: {item.raw_item.name[:100]}{item.raw_item.arguments[:100]}...")
                    elif hasattr(item.raw_item, 'role'):
                        print(f"  {item.raw_item.role}: {item.raw_item.status[:100]}")
                else:
                    pass

    else:
        print("No raw responses available to show inner dialog.")


async def run_test_cases(triage_assistant):
    """Run predefined test cases for the triage assistant.
    
    Args:
        triage_assistant: An Assistant instance or an OpenAI Agent instance
    """
    start_time = time.time()
    
    # Get the agent to use for tests
    if hasattr(triage_assistant, 'get_agent'):
        # If it's an Assistant, get its agent
        agent_to_use = triage_assistant.get_agent()
    else:
        # If it's already an OpenAI Agent
        agent_to_use = triage_assistant

    # Test Case 1: Date request - should be handled by Date Assistant
    print("--- Test Case 1: Date Request ---")
    result_date_request = await Runner.run(
        agent_to_use,
        input="Hello, what is the current date?",
    )

    # Print inner dialog
    print_agent_inner_dialog(result_date_request)

    total_tokens_date_request = sum_total_tokens(result_date_request.raw_responses)
    print(f"\nTotal Tokens Used for Date Request: {total_tokens_date_request}")
    print(f"\nFinal response (Date Request): {result_date_request.final_output}")
    print(f"\nAgent responsible: {result_date_request.last_agent.name}")

    print("\n" + format_separator_line() + "\n")

    # Test Case 2: Time request - should be handled by Time Assistant
    print("--- Test Case 2: Time Request ---")
    result_time_request = await Runner.run(
        agent_to_use,
        input="What time is it right now?",
    )

    # Print inner dialog
    print_agent_inner_dialog(result_time_request)

    total_tokens_time_request = sum_total_tokens(result_time_request.raw_responses)
    print(f"\nTotal Tokens Used for Time Request: {total_tokens_time_request}")
    print(f"\nFinal response (Time Request): {result_time_request.final_output}")
    print(f"\nAgent responsible: {result_time_request.last_agent.name}")

    print("\n" + format_separator_line() + "\n")

    # Test Case 3: Coding request - should be handled by Code Assistant
    print("--- Test Case 3: Coding Request ---")
    result_code_request = await Runner.run(
        agent_to_use,
        input="Write a Python function to calculate the Fibonacci sequence, save it to a file underneath the current directory 'build', and execute it.",
    )

    # Print inner dialog
    print_agent_inner_dialog(result_code_request)

    total_tokens_code_request = sum_total_tokens(result_code_request.raw_responses)
    print(f"\nTotal Tokens Used for Code Request: {total_tokens_code_request}")
    print(f"\nFinal response (Code Request): {result_code_request.final_output}")
    print(f"\nAgent responsible: {result_code_request.last_agent.name}")

    print("\n" + format_separator_line() + "\n")

    # Test Case 4: Writing request - should be handled by Writing Assistant
    print("--- Test Case 4: Writing Request ---")
    result_writing_request = await Runner.run(
        agent_to_use,
        input="Help me write a professional email to request a meeting with a client",
    )

    # Print inner dialog
    print_agent_inner_dialog(result_writing_request)

    total_tokens_writing_request = sum_total_tokens(result_writing_request.raw_responses)
    print(f"\nTotal Tokens Used for Writing Request: {total_tokens_writing_request}")
    print(f"\nFinal response (Writing Request): {result_writing_request.final_output}")
    print(f"\nAgent responsible: {result_writing_request.last_agent.name}")

    print("\n" + format_separator_line() + "\n")

    # Test Case 5: Math request - should be handled by Math Assistant
    print("--- Test Case 5: Math Request ---")
    result_math_request = await Runner.run(
        agent_to_use,
        input="Calculate the integral of x^2 with respect to x",
    )

    # Print inner dialog
    print_agent_inner_dialog(result_math_request)

    total_tokens_math_request = sum_total_tokens(result_math_request.raw_responses)
    print(f"\nTotal Tokens Used for Math Request: {total_tokens_math_request}")
    print(f"\nFinal response (Math Request): {result_math_request.final_output}")
    print(f"\nAgent responsible: {result_math_request.last_agent.name}")

    print("\n" + format_separator_line() + "\n")

    # Test Case 6: Knowledge Request - should be handled by the Knowledge Assistant
    print("--- Test Case 6: Knowledge Request ---")
    result_general_request = await Runner.run(
        agent_to_use,
        input="Hello, tell me something about MAX4 in Lund, Sweden",
    )

    # Print inner dialog
    print_agent_inner_dialog(result_general_request)

    total_tokens_general_request = sum_total_tokens(result_general_request.raw_responses)
    print(f"\nTotal Tokens Used for General Request: {total_tokens_general_request}")
    print(f"\nFinal response (General Request): {result_general_request.final_output}")
    print(f"\nAgent responsible: {result_general_request.last_agent.name}")

    print("\n" + format_separator_line() + "\n")

    # Test Case 7: General request - should be handled by Triage Agent directly
    print("--- Test Case 7: General Request ---")
    result_knowledge_request = await Runner.run(
        agent_to_use,
        input="Hello, how can you help me today?",
    )

    # Print inner dialog
    print_agent_inner_dialog(result_knowledge_request)

    total_tokens_knowledge_request = sum_total_tokens(result_knowledge_request.raw_responses)
    print(f"\nTotal Tokens Used for General Request: {total_tokens_knowledge_request}")
    print(f"\nFinal response (General Request): {result_knowledge_request.final_output}")
    print(f"\nAgent responsible: {result_knowledge_request.last_agent.name}")

    print("\n" + format_separator_line() + "\n")

    total_tokens_all = (
        total_tokens_date_request +
        total_tokens_time_request +
        total_tokens_code_request +
        total_tokens_writing_request +
        total_tokens_math_request +
        total_tokens_general_request +
        total_tokens_knowledge_request
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal Tokens Used for All Tests: {total_tokens_all}")
    print(f"Total Time Taken for All Tests: {elapsed_time:.2f} seconds")

    print("\n" + format_separator_line() + "\n")