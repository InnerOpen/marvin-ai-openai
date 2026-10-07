"""Credentials, prices and models: what Marvin reads to build, bill and offer the providers."""

import pytest
from marvin_integration_sdk.ai import AIConfigError, masked_credentials, read_credentials

from marvin_ai_openai import AzureOpenAIProvider, OpenAIProvider


def test_openai_reads_the_settings_core_reads():
    assert [c.env("openai") for c in OpenAIProvider.credentials] == ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
    built = OpenAIProvider.from_credentials(read_credentials(OpenAIProvider.credentials, {"api_key": "test-key"}))
    assert built._base_url is None and built._responses is True


def test_azure_needs_an_endpoint_and_defaults_the_api_version():
    assert [c.env("azure") for c in AzureOpenAIProvider.credentials] == ["AZURE_API_KEY", "AZURE_BASE_URL", "AZURE_API_VERSION"]
    with pytest.raises(AIConfigError, match="base_url"):
        read_credentials(AzureOpenAIProvider.credentials, {"api_key": "test-key"})
    built = AzureOpenAIProvider.from_credentials(
        read_credentials(AzureOpenAIProvider.credentials, {"api_key": "k", "base_url": "https://x.openai.azure.com"})
    )
    assert built._api_version == "2024-02-01"


def test_the_key_is_masked():
    assert masked_credentials(OpenAIProvider.credentials, {"api_key": "test-key", "base_url": None}) == {"api_key": "****", "base_url": None}


def test_prices_match_what_core_charged():
    # The table Marvin core's pricing.py had for "openai" on 2026-10-07; the plugin must not change a bill.
    assert OpenAIProvider.estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75
    assert OpenAIProvider.estimate_cost("gpt-4o-2024-08-06", 1_000_000, 0) == 2.5
    assert OpenAIProvider.estimate_cost("gpt-image-1", 0, 1_000_000) == 40.0
    assert OpenAIProvider.estimate_cost("gpt-5.6-luna", 1_000_000, 0) == 0.2
    assert OpenAIProvider.estimate_cost("gpt-99", 10, 10) is None
    assert AzureOpenAIProvider.estimate_cost("gpt-4o", 1_000_000, 0) == 2.5


def test_the_picker_suggests_chat_models_only():
    assert OpenAIProvider.default_model in OpenAIProvider.suggested_models
    assert not any("image" in m for m in OpenAIProvider.suggested_models)
