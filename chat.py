"""
Chat interfaces for the PyMultiAgent framework.

This module provides abstract and concrete chat interface implementations
for interacting with assistants.
"""
from abc import ABC, abstractmethod
from openai.types.responses import ResponseTextDeltaEvent, ResponseFunctionToolCall
from agents import ItemHelpers

class AbstractChat(ABC):
    """
    Abstract base class for chat interfaces.
    Provides a common interface for different chat implementations.
    """
    
    def __init__(self, assistant, max_turns=15):
        """
        Initialize the chat interface.
        
        Args:
            assistant: The initialized assistant to use for the chat
            max_turns (int): Maximum turns allowed for each agent run
        """
        self.assistant = assistant
        self.max_turns = max_turns
        self.chat_history = []
    
    @abstractmethod
    async def run(self):
        """
        Run the chat session with the user.
        This is the main entry point that orchestrates the chat flow.
        """
        pass


class CLIChat(AbstractChat):
    """
    Command-line interface implementation of the abstract chat interface.
    """
    
    async def run(self):
        """
        Run the chat session with the user.
        This is the main entry point that orchestrates the chat flow.
        """
        await self._display_welcome()
        
        while True:
            user_input, continue_chat = await self._get_user_input()
            if not continue_chat:
                break
                
            # Check for special commands
            if await self._handle_special_commands(user_input):
                continue
                
            # Add user input to chat history
            self.chat_history = self.chat_history + [{"role": "user", "content": user_input}]
            
            print("Processing your request (streaming)...")
            
            try:
                # Process the user input and get a response
                await self._process_request()
            except Exception as e:
                await self._display_error(str(e))
                # Remove the last user message if there was an error
                if self.chat_history:
                    self.chat_history.pop()
    
    async def _get_user_input(self):
        """
        Get input from the user via the command line.
        
        Returns:
            str: The user input
            bool: True if the chat should continue, False if it should exit
        """
        prompt = input("\nEnter the next prompt (or 'exit'): ")
        
        # Check if the user wants to exit
        if prompt.lower() == "exit":
            print("Exiting chat mode. Goodbye!")
            return prompt, False
            
        return prompt, True
    
    async def _display_response(self, response, agent_name=None):
        """
        Display a response to the user on the command line.
        
        Args:
            response (str): The response text
            agent_name (str, optional): The name of the agent that generated the response
        """
        if agent_name:
            print(f"  Agent: {agent_name}", flush=True)
        print(response, end="", flush=True)
    
    async def _handle_special_commands(self, user_input):
        """
        Handle special commands like 'help', 'clear', etc.
        
        Args:
            user_input (str): The user input to check for commands
            
        Returns:
            bool: True if a special command was handled, False otherwise
        """
        if user_input.lower() == "help":
            print("\nAvailable commands:")
            print("  exit - End the conversation")
            print("  help - Show this help message")
            print("  clear - Clear the chat history")
            return True
        elif user_input.lower() == "clear":
            self.chat_history = []
            print("Chat history cleared.")
            return True
            
        return False
    
    async def _process_request(self):
        """
        Process the user request and update the chat history.
        This method handles the interaction with the assistant.
        """
        streamed_output = ""
        last_agent = None
        
        # Using the Assistant's run_streamed method
        stream = self.assistant.run_streamed(
            input_messages=self.chat_history,
            max_turns=self.max_turns
        )
        
        async for event in stream.stream_events():
            streamed_output, last_agent = await self._handle_stream_event(event, streamed_output, last_agent)
        
        print()  # Newline after streaming output
                
        # Update chat history with the streamed response
        if streamed_output:
            self.chat_history = self.chat_history + [{"role": "assistant", "content": streamed_output}]
    
    async def _handle_stream_event(self, event, streamed_output, last_agent):
        """
        Handle a streaming event from the assistant on the command line.
        
        Args:
            event: The stream event
            streamed_output (str): The current streamed output text
            last_agent: The last agent that generated output
            
        Returns:
            tuple: (updated_streamed_output, updated_last_agent)
        """
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
                streamed_output += event.data.delta
        elif event.type == "agent_updated_stream_event":
            last_agent = event.new_agent
            print(f"  Agent updated: {last_agent.name}", flush=True)
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                if hasattr(event.item, "raw_item") and isinstance(event.item.raw_item, ResponseFunctionToolCall):
                    tool_call = event.item.raw_item
                    print(f"  Tool call: {tool_call.name}{tool_call.arguments[:100]}...", flush=True)
                else:
                    print("  Tool was called", flush=True)
            elif event.item.type == "tool_call_output_item":
                print(f"  Tool output: {event.item.output}", flush=True)
            elif event.item.type == "message_output_item":
                text = ItemHelpers.text_message_output(event.item)
                # If no token output has been printed, use this complete text as final output.
                if not streamed_output:
                    streamed_output = text
                    
        return streamed_output, last_agent
    
    async def _display_welcome(self):
        """
        Display a welcome message on the command line.
        """
        print("--- Interactive Chat Mode ---")
        print("Type 'exit' to end the conversation")
        print("Type 'help' for available commands")
    
    async def _display_error(self, error_message):
        """
        Display an error message on the command line.
        
        Args:
            error_message (str): The error message to display
        """
        print(f"\nError processing request: {error_message}")
        print("Your message was not added to the chat history.")