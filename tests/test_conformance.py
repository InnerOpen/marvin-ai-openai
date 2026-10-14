"""The SDK's AI provider conformance kit, over a fake OpenAI server, for each way the plugin talks:
the official API (Responses), an OpenAI-compatible server (Chat Completions) and Azure OpenAI."""

import pytest
from fake_server import FakeOpenAIServer
from marvin_integration_sdk.ai.testing import AIProviderContract

from marvin_ai_openai import AzureOpenAIProvider, OpenAIProvider, azure_plugin, plugin
from marvin_ai_openai.openai_api import _learned

PLACEHOLDER_KEY = "test-key"


@pytest.fixture(autouse=True)
def _forget():
    _learned.clear()
    yield
    _learned.clear()


class TestOpenAI(AIProviderContract):
    @pytest.fixture
    def transport(self):
        return FakeOpenAIServer()

    @pytest.fixture
    def provider(self, transport):
        return OpenAIProvider(api_key=PLACEHOLDER_KEY, http_client=transport.client(), max_retries=0)

    def test_it_talks_to_the_responses_api(self, provider, transport, model):
        from marvin_integration_sdk.ai import Message

        transport.reply_text("hi")
        provider.complete([Message("user", "hi")], model)
        assert transport.requests[-1].url.path == "/v1/responses"
        assert transport.last_json()["store"] is False


class TestOpenAICompatibleServer(AIProviderContract):
    @pytest.fixture
    def transport(self):
        return FakeOpenAIServer()

    @pytest.fixture
    def provider(self, transport):
        return OpenAIProvider(api_key=PLACEHOLDER_KEY, base_url="http://localhost:8000/v1", http_client=transport.client(), max_retries=0)

    def test_it_talks_to_chat_completions(self, provider, transport, model):
        from marvin_integration_sdk.ai import Message

        transport.reply_text("hi")
        provider.complete([Message("user", "hi")], model)
        assert transport.requests[-1].url.path == "/v1/chat/completions"


class TestAzureOpenAI(AIProviderContract):
    @pytest.fixture
    def transport(self):
        return FakeOpenAIServer()

    @pytest.fixture
    def provider(self, transport):
        return AzureOpenAIProvider(
            api_key=PLACEHOLDER_KEY, base_url="https://example.openai.azure.com", http_client=transport.client(), max_retries=0
        )

    @pytest.fixture
    def model(self):
        return "my-gpt-4o-deployment"

    def test_it_calls_the_deployment(self, provider, transport, model):
        from marvin_integration_sdk.ai import Message

        transport.reply_text("hi")
        provider.complete([Message("user", "hi")], model)
        request = transport.requests[-1]
        assert request.url.path == "/openai/deployments/my-gpt-4o-deployment/chat/completions"
        assert request.url.params["api-version"] == "2024-02-01"


def test_the_entry_points_declare_both_plugins():
    from importlib.metadata import entry_points

    eps = {ep.name: ep.load() for ep in entry_points(group="marvin.ai_providers")}
    assert eps["openai"] is plugin and eps["azure"] is azure_plugin


def test_cached_prompt_tokens_come_from_the_usage_details():
    """OpenAI's automatic cache reports what it served inside the prompt count (input_tokens_details.cached_tokens)."""
    from marvin_integration_sdk.ai import Message

    transport = FakeOpenAIServer()
    transport.reply_text("ok", prompt_tokens=30000, completion_tokens=4, cached=27000)
    provider = OpenAIProvider(api_key=PLACEHOLDER_KEY, http_client=transport.client(), max_retries=0)
    result = provider.complete([Message("user", "hi")], "gpt-5.4-mini")
    assert (result.prompt_tokens, result.cached_prompt_tokens, result.cache_write_tokens) == (30000, 27000, 0)
