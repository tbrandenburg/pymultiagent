#!/usr/bin/env uv run python
"""
Test script for the backends module.

This script tests the Backend registry and the various backend implementations.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

# Import the backends module
try:
    from backends import (
        Backends,
        Backend,
        AzureBackend,
        OpenAIBackend,
        LlamaBackend,
        initialize_backend_types,
        configure_backends,
        get_chat_model
    )
except ImportError:
    try:
        # Try relative import if direct import fails
        from .backends import (
            Backends,
            Backend,
            AzureBackend,
            OpenAIBackend,
            LlamaBackend,
            initialize_backend_types,
            configure_backends,
            get_chat_model
        )
    except ImportError:
        # If all else fails, inform the user
        raise ImportError("Could not import backends module. Make sure you're running from the correct directory.")


class TestBackends(unittest.TestCase):
    """Tests for the Backend registry and implementations."""
    
    def setUp(self):
        """Reset the Backends registry before each test."""
        # Clear the backends registry
        Backends._backends = {}
        # Initialize backend types
        initialize_backend_types()
    
    def test_backend_registration(self):
        """Test that backends are properly registered."""
        # Check that all expected backends are registered
        registered_types = Backends.get_registered_types()
        self.assertIn("azure", registered_types)
        self.assertIn("openai", registered_types)
        self.assertIn("llama", registered_types)
        
        # Check that we can get each backend
        azure_backend = Backends.get("azure")
        openai_backend = Backends.get("openai")
        llama_backend = Backends.get("llama")
        
        self.assertIsInstance(azure_backend, AzureBackend)
        self.assertIsInstance(openai_backend, OpenAIBackend)
        self.assertIsInstance(llama_backend, LlamaBackend)
    
    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "test-azure-key",
        "AZURE_OPENAI_ENDPOINT": "https://test-azure-endpoint",
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_API_ENDPOINT": "https://test-openai-endpoint",
        "LLAMA_OPENAI_API_KEY": "test-llama-key",
        "LLAMA_OPENAI_ENDPOINT": "http://test-llama-endpoint",
    })
    def test_backend_configuration(self):
        """Test that backends can be configured with API keys and endpoints."""
        # Configure backends
        configure_backends()
        
        # Check Azure configuration
        azure_backend = Backends.get("azure")
        self.assertEqual(azure_backend.api_key, "test-azure-key")
        self.assertEqual(azure_backend.endpoint, "https://test-azure-endpoint")
        self.assertIn("gpt-4o", azure_backend.get_models())
        self.assertIn("o4-mini", azure_backend.get_models())
        
        # Check OpenAI configuration
        openai_backend = Backends.get("openai")
        self.assertEqual(openai_backend.api_key, "test-openai-key")
        self.assertEqual(openai_backend.endpoint, "https://test-openai-endpoint")
        self.assertIn("gpt-4", openai_backend.get_models())
        self.assertIn("gpt-3.5-turbo", openai_backend.get_models())
        
        # Check Llama configuration
        llama_backend = Backends.get("llama")
        self.assertEqual(llama_backend.api_key, "test-llama-key")
        self.assertEqual(llama_backend.endpoint, "http://test-llama-endpoint")
        self.assertIn("llama3.3", llama_backend.get_models())
    
    @patch('backends.OpenAIChatCompletionsModel')
    @patch('backends.AsyncOpenAI')
    def test_get_chat_model(self, mock_async_openai, mock_chat_model):
        """Test that get_chat_model returns a properly configured model."""
        # Set up mocks
        mock_client = MagicMock()
        mock_async_openai.return_value = mock_client
        mock_model = MagicMock()
        mock_chat_model.return_value = mock_model
        
        # Configure a backend
        openai_backend = Backends.get("openai")
        openai_backend.configure("test-key", "https://test-endpoint")
        openai_backend.add_model("gpt-4", {})
        
        # Get a chat model
        model = get_chat_model("openai", "gpt-4")
        
        # Verify correct client creation
        mock_async_openai.assert_called_once()
        self.assertEqual(mock_async_openai.call_args.kwargs["api_key"], "test-key")
        
        # Verify correct model creation
        mock_chat_model.assert_called_once()
        self.assertEqual(mock_chat_model.call_args.kwargs["openai_client"], mock_client)
        self.assertEqual(mock_chat_model.call_args.kwargs["model"], "gpt-4")
        self.assertEqual(model, mock_model)


class TestCustomBackend(unittest.TestCase):
    """Tests for creating and using custom backends."""
    
    def setUp(self):
        """Reset the Backends registry before each test."""
        # Clear the backends registry
        Backends._backends = {}
    
    def test_custom_backend(self):
        """Test creating and using a custom backend."""
        # Create a custom backend class
        class CustomBackend(Backend):
            def __init__(self):
                super().__init__("custom")
                self.api_key = None
                self.custom_param = None
            
            def configure(self, api_key, custom_param):
                self.api_key = api_key
                self.custom_param = custom_param
                return self
            
            def create_client(self, model_name):
                if model_name not in self.models:
                    raise ValueError(f"Model {model_name} not registered")
                # Return a mock client for testing
                mock = MagicMock()
                # Make the mock have the expected attributes for OpenAI clients
                mock.base_url = "https://api.custom.com"
                return mock
        
        # Register the custom backend
        custom_backend = CustomBackend()
        Backends.register(custom_backend)
        
        # Configure the backend
        custom_backend.configure("custom-key", "custom-param")
        custom_backend.add_model("custom-model", {"param": "value"})
        
        # Verify registration
        self.assertIn("custom", Backends.get_registered_types())
        self.assertEqual(Backends.get("custom"), custom_backend)
        
        # Verify configuration
        self.assertEqual(custom_backend.api_key, "custom-key")
        self.assertEqual(custom_backend.custom_param, "custom-param")
        self.assertIn("custom-model", custom_backend.get_models())
        self.assertEqual(custom_backend.models["custom-model"], {"param": "value"})
        
        # Test client creation
        client = custom_backend.create_client("custom-model")
        self.assertIsNotNone(client)


if __name__ == "__main__":
    unittest.main()