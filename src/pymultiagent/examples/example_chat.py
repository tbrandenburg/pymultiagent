#!/usr/bin/env python
"""
Example script demonstrating how to use the chat interfaces.
"""
import asyncio
import argparse
from pymultiagent.agents import Agent
from pymultiagent.chat import CLIChat

async def main():
    """
    Main function demonstrating the use of CLIChat.
    """
    parser = argparse.ArgumentParser(description="Example chat script")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--backend", default="azure", help="Backend to use")
    parser.add_argument("--max_turns", type=int, default=15, help="Maximum conversation turns")
    args = parser.parse_args()

    # Create a simple agent
    agent = Agent(
        name="Simple Agent",
        instructions=(
            "You are a friendly and helpful agent. "
            "Answer questions concisely and accurately."
        ),
        backend=args.backend,
        model_name=args.model
    )

    print(f"Starting chat with {args.backend} backend and {args.model} model")

    # Create and run the CLI chat interface
    chat = CLIChat(agent, max_turns=args.max_turns)
    await chat.run()

if __name__ == "__main__":
    asyncio.run(main())
