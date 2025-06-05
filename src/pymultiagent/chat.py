"""
Chat interfaces for the PyMultiAgent framework.

This module provides abstract and concrete chat interface implementations
for interacting with assistants.
"""
from abc import ABC, abstractmethod
import asyncio
import logging
from typing import Dict, List, Any
from openai.types.responses import ResponseTextDeltaEvent, ResponseFunctionToolCall

from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram.request import HTTPXRequest
from telegram.constants import ParseMode

# Get module logger
logger = logging.getLogger(__name__)

# Set HTTPX and telegram.ext loggers to WARNING level to reduce API call logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)

# Define a function to extract text from message items
def extract_message_text(item):
    """
    Extract text from a message output item.
    Returns the content if available.
    """
    if hasattr(item, 'message') and hasattr(item.message, 'content'):
        return item.message.content
    return ""

# Configure logging
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
                # Extract text from the message
                text = extract_message_text(event.item)

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


class TelegramChat(AbstractChat):
    """
    Telegram interface implementation of the abstract chat interface.
    This class provides a chat interface for interacting with assistants via Telegram.
    """

    def __init__(self, assistant, token, max_turns=15, allowed_user_ids=None, verify_ssl=True):
        """
        Initialize the Telegram chat interface.

        Args:
            assistant: The initialized assistant to use for the chat
            token (str): Telegram Bot API token
            max_turns (int): Maximum turns allowed for each agent run
            allowed_user_ids (list, optional): List of user IDs allowed to interact with the bot
            verify_ssl (bool, optional): Whether to verify SSL certificates. Set to False to bypass SSL errors.
        """
        super().__init__(assistant, max_turns)
        self.token = token
        self.allowed_user_ids = allowed_user_ids or []
        self.user_sessions: Dict[int, List[Dict[str, Any]]] = {}  # Store chat history per user
        self.active_requests: Dict[int, asyncio.Task] = {}  # Track active requests per user
        self.verify_ssl = verify_ssl
        self.bot = None

    async def run(self):
        """
        Run the Telegram bot to handle chat sessions with users.
        This is the main entry point that starts the Telegram bot.
        """

        # Setup bot with SSL verification settings
        application_builder = Application.builder().token(self.token)

        # Configure SSL verification
        if not self.verify_ssl:
            logger.warning("SSL certificate verification is disabled. This is insecure!")
            # Create a custom request with SSL verification disabled
            request = HTTPXRequest(connection_pool_size=256, httpx_kwargs={"verify": False})
            application_builder = application_builder.request(request)

        # Setup bot handlers
        self.application = application_builder.build()

        # Add handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("clear", self._clear_command))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))

        # Start the bot in a proper async way
        logger.info("Starting Telegram bot...")

        try:
            # Initialize the application (doesn't start the event loop)
            await self.application.initialize()

            # Start the application - this just prepares it but doesn't fetch updates
            await self.application.start()

            # Now start the updater to actually fetch updates from Telegram
            # We need to explicitly start the updater to get messages
            if not self.application.updater:
                logger.error("Updater not available. Cannot start polling.")
                return

            # Start polling for updates - this is what actually connects to Telegram
            await self.application.updater.start_polling(
                allowed_updates=["message", "edited_message", "callback_query", "chat_member"],
                drop_pending_updates=True,  # Ignore any updates that occurred while the bot was offline
                poll_interval=2.0,  # Reduce polling frequency to minimize logs and API calls
                timeout=10  # Longer timeout for long polling
            )
            logger.info("Telegram bot successfully started and is polling for updates")

            # Keep the bot running until stopped externally
            # Use asyncio.Event as a simple way to keep the coroutine running
            stop_event = asyncio.Event()
            try:
                # Wait indefinitely for an event that never gets set
                await stop_event.wait()
            except asyncio.CancelledError:
                # Handle graceful shutdown if the coroutine is cancelled
                logger.info("Telegram bot stopping due to cancellation...")
            finally:
                # Clean up when exiting - stop updater first, then the application
                logger.info("Telegram bot stopping and shutting down...")
                if self.application.updater:
                    await self.application.updater.stop()
                await self.application.stop()
                await self.application.shutdown()
        except Exception as e:
            logger.error(f"Error in Telegram bot: {e}")
            raise

    async def _start_command(self, update: 'Any', context: 'Any'):
        """
        Handle the /start command.

        Args:
            update: The update object from Telegram
            context: The context object from Telegram
        """
        user_id = update.effective_user.id

        # Check if user is allowed to use the bot
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            await update.message.reply_text("You are not authorized to use this bot.", parse_mode=None)
            return await asyncio.sleep(0)  # Return an awaitable

        # Initialize user session if not exists
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []

        await update.message.reply_text(
            "Welcome to the PyMultiAgent Telegram bot!\n\n"
            "I'm powered by advanced AI to assist you with various tasks.\n\n"
            "Available commands:\n"
            "/help - Show this help message\n"
            "/clear - Clear your chat history\n\n"
            "Just send me a message to get started!",
            parse_mode="Markdown"
        )
        return await asyncio.sleep(0)  # Return an awaitable

    async def _help_command(self, update: 'Any', context: 'Any'):
        """
        Handle the /help command.

        Args:
            update: The update object from Telegram
            context: The context object from Telegram
        """
        await update.message.reply_text(
            "Available commands:\n"
            "/start - Start or restart the bot\n"
            "/help - Show this help message\n"
            "/clear - Clear your chat history\n\n"
            "Just send me a message to get started!",
            parse_mode="Markdown"
        )
        return await asyncio.sleep(0)  # Return an awaitable

    async def _clear_command(self, update: 'Any', context: 'Any'):
        """
        Handle the /clear command to clear chat history.

        Args:
            update: The update object from Telegram
            context: The context object from Telegram
        """
        user_id = update.effective_user.id

        # Check if user is allowed to use the bot
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            await update.message.reply_text("You are not authorized to use this bot.", parse_mode=None)
            return await asyncio.sleep(0)  # Return an awaitable

        # Clear user session
        self.user_sessions[user_id] = []
        await update.message.reply_text("Chat history cleared.", parse_mode="Markdown")
        return await asyncio.sleep(0)  # Return an awaitable

    async def _handle_message(self, update: 'Any', context: 'Any'):
        """
        Handle incoming text messages.

        Args:
            update: The update object from Telegram
            context: The context object from Telegram
        """
        user_id = update.effective_user.id
        user_message = update.message.text

        logger.debug(f"{user_id}: {user_message}")

        # Check if user is allowed to use the bot
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            await update.message.reply_text("You are not authorized to use this bot.")
            return await asyncio.sleep(0)  # Return an awaitable

        # Initialize user session if not exists
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []

        # Check if a request is already active for this user
        if user_id in self.active_requests and not self.active_requests[user_id].done():
            await update.message.reply_text("I'm still processing your previous request. Please wait a moment.", parse_mode=None)
            return await asyncio.sleep(0)  # Return an awaitable

        # Add user message to chat history
        self.user_sessions[user_id].append({"role": "user", "content": user_message})

        # Create and start a task to process the request
        self.active_requests[user_id] = asyncio.create_task(
            self._process_telegram_request(update, user_id)
        )
        return await asyncio.sleep(0)  # Return an awaitable

    async def _process_telegram_request(self, update, user_id):
        """
        Process a user request from Telegram.

        Args:
            update: The update object from Telegram
            user_id: The Telegram user ID
        """
        try:
            # Send typing indicator
            await update.message.chat.send_action("typing")

            # Set the current user's chat history
            self.chat_history = self.user_sessions[user_id]

            # Process the request
            response_text = await self._get_assistant_response()

            # Add assistant response to chat history
            self.user_sessions[user_id].append({"role": "assistant", "content": response_text})

            # Process code blocks for proper Markdown formatting in Telegram
            # Telegram uses different markdown syntax than the standard markdown
            response_text = self._prepare_telegram_markdown(response_text)

            # Send the response in chunks if it's too long, using Markdown formatting
            MAX_LENGTH = 4096
            for i in range(0, len(response_text), MAX_LENGTH):
                chunk = response_text[i:i + MAX_LENGTH]
                await update.message.reply_text(
                    chunk,
                    parse_mode=ParseMode.MARKDOWN_V2
                )

        except Exception as e:
            error_message = str(e)
            logger.exception(f"Error processing Telegram request: {error_message}")
            await update.message.reply_text(f"Error processing your request: {error_message}", parse_mode=None)

            # Remove the last user message if there was an error
            if self.user_sessions[user_id]:
                self.user_sessions[user_id].pop()

    def _prepare_telegram_markdown(self, text):
        """
        Prepare text for Telegram's MarkdownV2 format.

        Telegram's MarkdownV2 requires escaping of special characters.
        This function ensures that formatting works correctly in Telegram.

        Args:
            text: The text to prepare

        Returns:
            Properly formatted text for Telegram MarkdownV2
        """
        # Replace triple backticks with single backticks for code blocks
        # Telegram doesn't support language specification or path specification in code blocks
        import re

        # First, handle code blocks with path specification
        # Pattern matches ```path/to/file.ext#L123-456 ... ``` format
        path_code_pattern = r"```([^\n]+?#L\d+-\d+)\n(.*?)```"
        text = re.sub(path_code_pattern, r"*File: \1*\n`\2`", text, flags=re.DOTALL)

        # Then handle regular code blocks
        code_block_pattern = r"```(?:[a-zA-Z]*\n)?(.*?)```"
        text = re.sub(code_block_pattern, r"`\1`", text, flags=re.DOTALL)

        # Split the text into parts where code blocks (content between backticks) are separated
        parts = []
        current_pos = 0

        # Find all code blocks (text between backticks)
        for match in re.finditer(r'`(.*?)`', text, flags=re.DOTALL):
            # Add the text before the code block (with escaping)
            if match.start() > current_pos:
                normal_text = text[current_pos:match.start()]
                # Escape special characters in normal text
                normal_text = self._escape_markdown_v2_chars(normal_text)
                parts.append(normal_text)

            # Add the code block (without escaping inside it)
            code_block = text[match.start():match.end()]
            parts.append(code_block)
            current_pos = match.end()

        # Add any remaining text after the last code block
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            remaining_text = self._escape_markdown_v2_chars(remaining_text)
            parts.append(remaining_text)

        text = ''.join(parts)

        return text

    def _escape_markdown_v2_chars(self, text):
        """
        Escape special characters for MarkdownV2 format.

        Args:
            text: Text to escape

        Returns:
            Text with special characters escaped
        """
        # Characters that need to be escaped in MarkdownV2
        special_chars = '_*[]()~`>#+-=|{}.!'

        # Escape each special character with a backslash
        for char in special_chars:
            text = text.replace(char, '\\' + char)

        return text

    async def _get_assistant_response(self):
        """
        Get a response from the assistant.

        Returns:
            str: The assistant's response text
        """
        response_text = ""

        # Run the assistant with the current chat history
        stream = self.assistant.run_streamed(
            input_messages=self.chat_history,
            max_turns=self.max_turns
        )

        try:
            async for event in stream.stream_events():
                if event.type == "raw_response_event":
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        response_text += event.data.delta
                elif event.type == "run_item_stream_event":
                    if event.item.type == "message_output_item":
                        # Extract text from the message
                        text = extract_message_text(event.item)

                        # If no token output has been received, use the complete text
                        if not response_text:
                            response_text = text
        except Exception as e:
            logger.exception(f"Error in streaming response: {str(e)}")
            if not response_text:
                response_text = f"An error occurred while generating a response: {str(e)}"

        return response_text
