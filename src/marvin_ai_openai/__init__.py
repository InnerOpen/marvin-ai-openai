"""OpenAI for Marvin: the OpenAI API (and OpenAI-compatible servers) and Azure OpenAI, as AI provider plugins.

Two entry points in ``marvin.ai_providers``: ``openai`` and ``azure``. Each replaces Marvin core's
built-in provider of the same slug, and reads the same settings (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``;
``AZURE_API_KEY``, ``AZURE_BASE_URL``, ``AZURE_API_VERSION``) or a workspace's own key.
"""

from marvin_integration_sdk.ai import AIProviderPlugin

from .azure import AzureOpenAIProvider
from .provider import OpenAIProvider

__all__ = ["AzureOpenAIProvider", "OpenAIProvider", "azure_plugin", "plugin"]
__version__ = "0.1.0"

plugin = AIProviderPlugin(
    slug="openai", name="OpenAI", provider=OpenAIProvider, description="OpenAI's API (Responses), or an OpenAI-compatible server."
)
azure_plugin = AIProviderPlugin(
    slug="azure", name="Azure OpenAI", provider=AzureOpenAIProvider, description="OpenAI models deployed in an Azure resource."
)
