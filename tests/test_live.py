"""One optional smoke test against the real OpenAI API. Skipped unless OPENAI_API_KEY is set; CI has none."""

import os

import pytest
from marvin_integration_sdk.ai import CompletionOptions, Message

from marvin_ai_openai import OpenAIProvider

pytestmark = pytest.mark.live


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
def test_a_real_completion():
    provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
    result = provider.complete(
        [Message("user", "Reply with the single word: towel")], os.environ.get("OPENAI_MODEL", "gpt-4o-mini"), CompletionOptions(max_tokens=16)
    )
    assert result.content.strip() and result.prompt_tokens > 0
