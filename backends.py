from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
import os

try:
    from agents import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI, AsyncAzureOpenAI
except ImportError:
    try:
        # Try absolute imports when being run directly
        from agents import OpenAIChatCompletionsModel
        from openai import AsyncOpenAI, AsyncAzureOpenAI
    except ImportError:
        raise ImportError("Required packages not found. Please install with 'pip install openai openai-agents'")

class Backend(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, backend_type: str):
        self.backend_type = backend_type
        self.models: Dict[str, Dict[str, Any]] = {}
    
    @abstractmethod
    def create_client(self, model_name):
        """Create a client for a specific model."""
        # This method should return an AsyncOpenAI or AsyncAzureOpenAI client
        # or raise an appropriate exception
        raise NotImplementedError("Subclasses must implement this method")
    
    def add_model(self, model_name: str, model_config: Optional[Dict[str, Any]] = None):
        """Add a supported model to this backend."""
        self.models[model_name] = model_config or {}
        return self
    
    def get_chat_model(self, model_name: str):
        """Get a chat model instance for a specific model."""
        client = self.create_client(model_name)
        if client:
            return OpenAIChatCompletionsModel(openai_client=client, model=model_name)
        return None
    
    def get_models(self) -> List[str]:
        """Get all models registered for this backend."""
        return list(self.models.keys())


class AzureBackend(Backend):
    """Azure OpenAI backend implementation."""
    
    def __init__(self):
        super().__init__("azure")
        self.api_key: Optional[str] = None
        self.endpoint: Optional[str] = None
    
    def configure(self, api_key: str, endpoint: str):
        """Configure the Azure backend."""
        self.api_key = api_key
        self.endpoint = endpoint
        return self
    
    def create_client(self, model_name):
        """Create an Azure OpenAI client for the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered for Azure backend")
        
        if not self.api_key or not self.endpoint:
            raise ValueError("Azure backend not fully configured: missing api_key or endpoint")
        
        model_config = self.models[model_name]
        api_version = model_config.get("api_version")
        deployment = model_config.get("deployment")
        
        if not api_version or not deployment:
            raise ValueError(f"Model {model_name} configuration incomplete: missing api_version or deployment")
        
        # Create headers dictionary with string keys and string values
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "api-key": self.api_key
        }
        
        # type: ignore comments tell the type checker to ignore these lines
        return AsyncAzureOpenAI(  # type: ignore
            api_key=self.api_key,
            api_version=api_version,
            azure_endpoint=self.endpoint,
            azure_deployment=deployment,
            default_headers=headers
        )


class OpenAIBackend(Backend):
    """OpenAI backend implementation."""
    
    def __init__(self):
        super().__init__("openai")
        self.api_key: Optional[str] = None
        self.endpoint: Optional[str] = None  # Optional, as OpenAI usually uses the default endpoint
    
    def configure(self, api_key: str, endpoint: Optional[str] = None):
        """Configure the OpenAI backend."""
        self.api_key = api_key
        self.endpoint = endpoint
        return self
    
    def create_client(self, model_name):
        """Create an OpenAI client for the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered for OpenAI backend")
        
        if not self.api_key:
            raise ValueError("OpenAI backend not fully configured: missing api_key")
        
        # Add base_url only if endpoint is specified
        if self.endpoint:
            return AsyncOpenAI(  # type: ignore
                api_key=self.api_key,
                base_url=self.endpoint
            )
        else:
            return AsyncOpenAI(  # type: ignore
                api_key=self.api_key
            )


class LlamaBackend(Backend):
    """Llama backend implementation."""
    
    def __init__(self):
        super().__init__("llama")
        self.api_key: Optional[str] = None
        self.endpoint: Optional[str] = None
    
    def configure(self, api_key: str, endpoint: str):
        """Configure the Llama backend."""
        self.api_key = api_key
        self.endpoint = endpoint
        return self
    
    def create_client(self, model_name):
        """Create a Llama client for the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered for Llama backend")
        
        if not self.api_key or not self.endpoint:
            raise ValueError("Llama backend not fully configured: missing api_key or endpoint")
        
        return AsyncOpenAI(  # type: ignore
            api_key=self.api_key,
            base_url=self.endpoint
        )


class Backends:
    """Registry for managing LLM backends."""
    
    # Static storage for backends
    _backends = {}
    
    @classmethod
    def register(cls, backend):
        """Register a backend instance."""
        cls._backends[backend.backend_type] = backend
        return backend
    
    @classmethod
    def get(cls, backend_type):
        """Get a registered backend by type."""
        if backend_type not in cls._backends:
            raise ValueError(f"Backend {backend_type} not registered")
        return cls._backends[backend_type]
    
    @classmethod
    def get_all(cls):
        """Get all registered backends."""
        return cls._backends
    
    @classmethod
    def get_registered_types(cls):
        """Get all registered backend types."""
        return list(cls._backends.keys())
    
    @classmethod
    def get_chat_model(cls, backend_type, model_name):
        """Get a chat model instance for a specific backend and model."""
        backend = cls.get(backend_type)
        return backend.get_chat_model(model_name)
    
    @classmethod
    def from_dict(cls, config_dict):
        """
        Configure backends from a dictionary similar to the current BACKEND_CONFIGS.
        
        This is a helper method to maintain compatibility with the previous config format.
        """
        for backend_type, config in config_dict.items():
            # Skip if backend type is not registered
            if backend_type not in cls._backends:
                continue
                
            backend = cls.get(backend_type)
            
            # Configure the backend (method will vary by backend type)
            if backend_type == "azure":
                backend.configure(
                    config.get("api_key"),
                    config.get("endpoint")
                )
            elif backend_type == "openai":
                backend.configure(
                    config.get("api_key"),
                    config.get("endpoint")
                )
            elif backend_type == "llama":
                backend.configure(
                    config.get("api_key"),
                    config.get("endpoint")
                )
            
            # Register models for this backend
            for model_name, model_config in config.get("models", {}).items():
                backend.add_model(model_name, model_config)
        
        return cls


def initialize_backend_types():
    """Initialize the Backends registry with specialized backend instances."""
    # Register Azure OpenAI backend
    Backends.register(AzureBackend())
    
    # Register OpenAI backend
    Backends.register(OpenAIBackend())
    
    # Register Llama backend
    Backends.register(LlamaBackend())


def configure_backends():
    """Configure backends with runtime values."""
    # Azure OpenAI
    azure_backend = Backends.get("azure")
    azure_backend.configure(
        os.getenv("AZURE_OPENAI_API_KEY", ""),
        os.getenv("AZURE_OPENAI_ENDPOINT", "https://openaiapimanagementxcse.azure-api.net")
    )
    
    # Add models to Azure
    azure_backend.add_model("gpt-4o", {
        "deployment": "gpt-4o",
        "api_version": "2025-01-01-preview"
    })
    azure_backend.add_model("gpt-4o-mini", {
        "deployment": "gpt-4o-mini",
        "api_version": "2025-01-01-preview"
    })
    azure_backend.add_model("o3-mini", {
        "deployment": "o3-mini",
        "api_version": "2025-01-01-preview"
    })
    azure_backend.add_model("o4-mini", {
        "deployment": "o4-mini",
        "api_version": "2025-01-01-preview"
    })
    
    # OpenAI
    openai_backend = Backends.get("openai")
    openai_backend.configure(
        os.getenv("OPENAI_API_KEY", ""),
        os.getenv("OPENAI_API_ENDPOINT", "https://api.openai.com/v1/")
    )
    
    # Add models to OpenAI
    openai_backend.add_model("gpt-3.5-turbo", {})
    openai_backend.add_model("gpt-4", {})
    openai_backend.add_model("gpt-4-turbo", {})
    openai_backend.add_model("gpt-4o", {})
    openai_backend.add_model("gpt-4o-mini", {})
    openai_backend.add_model("o3-mini", {})
    openai_backend.add_model("o4-mini", {})
    
    # Llama
    llama_backend = Backends.get("llama")
    llama_backend.configure(
        os.getenv("LLAMA_OPENAI_API_KEY", "dummy"),
        os.getenv("LLAMA_OPENAI_ENDPOINT", "http://localhost:8000/v1/")
    )
    
    # Add models to Llama
    llama_backend.add_model("llama3.2", {})
    llama_backend.add_model("llama3.3", {})


def get_chat_model(backend: str = "azure", model_name: str = "o4-mini"):
    """
    Returns an OpenAIChatCompletionsModel using the selected backend and model.
    Maintains backward compatibility with the previous design.
    """
    return Backends.get_chat_model(backend, model_name)


def load_backend_config_from_dict(config_dict):
    """
    Load backend configuration from a dictionary, similar to the old BACKEND_CONFIGS.
    This is a helper function for backward compatibility.
    """
    return Backends.from_dict(config_dict)