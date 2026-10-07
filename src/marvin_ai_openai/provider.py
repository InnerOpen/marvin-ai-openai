"""OpenAI, and any OpenAI-compatible server.

The official API goes through the Responses API (see openai_api); a custom base URL means an
OpenAI-compatible server, which gets Chat Completions — most of them don't implement Responses.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from marvin_integration_sdk.ai import (
    API_KEY,
    AIProvider,
    CompletionOptions,
    CompletionResult,
    Credential,
    Message,
    ToolDefinition,
)

from .openai_api import (
    TOOL_CHOICES,
    NeedsResponsesAPI,
    chat_messages,
    chat_result,
    chat_tools,
    create_chat_completion,
    create_response,
    json_schema_format,
    needs_responses,
    response_input,
    response_result,
    response_tools,
)
from .prices import OPENAI_PRICES

OFFICIAL_HOST = "api.openai.com"
# Model families the account lists that can't hold a conversation (embeddings, speech, images, moderation…).
NON_CHAT_MARKERS = ("embedding", "tts", "whisper", "transcribe", "dall-e", "image", "moderation", "audio", "realtime", "davinci", "babbage")


def openai_client(cls_name: str, **kwargs):
    try:
        import openai
    except ImportError:
        raise ImportError("The OpenAI SDK is not installed (pip install marvin-ai-openai brings it).") from None
    return getattr(openai, cls_name)(**kwargs)


class OpenAIProvider(AIProvider):
    provider_type = "openai"
    display_name = "OpenAI"
    credentials = (
        API_KEY,
        Credential("base_url", "Base URL", help="Leave empty for OpenAI. Set it for an OpenAI-compatible server, which gets Chat Completions."),
    )
    supports_vision = True
    supports_structured_output = True
    supports_embeddings = True
    supports_tool_calls = True
    prices = OPENAI_PRICES
    default_model = "gpt-4o-mini"
    suggested_models = tuple(m for m in OPENAI_PRICES if "image" not in m)
    default_embedding_model = "text-embedding-3-small"

    def __init__(self, api_key: str, base_url: str | None = None, **client_options: Any) -> None:
        """``client_options`` go to ``openai.OpenAI`` as they are (tests pass ``http_client`` and
        ``max_retries``)."""
        self._api_key = api_key
        self._base_url = base_url
        self._client_options = client_options
        self._responses = not base_url or OFFICIAL_HOST in base_url

    @classmethod
    def from_credentials(cls, values: Mapping[str, Any]) -> OpenAIProvider:
        return cls(api_key=values.get("api_key") or "", base_url=values.get("base_url") or None)

    def _client(self):
        return openai_client("OpenAI", api_key=self._api_key, base_url=self._base_url, **self._client_options)

    def complete(self, messages: list[Message], model: str, options: CompletionOptions | None = None) -> CompletionResult:
        opts = options or CompletionOptions()
        client = self._client()
        if self._responses:
            return response_result(create_response(client, model, opts, input=response_input(messages)))
        return chat_result(create_chat_completion(client, model, opts, messages=chat_messages(messages)))

    def complete_with_tools(
        self,
        messages: list[Message],
        model: str,
        tools: list[ToolDefinition],
        options: CompletionOptions | None = None,
        tool_choice: str = "auto",
    ) -> CompletionResult:
        opts = options or CompletionOptions()
        client = self._client()
        choice = TOOL_CHOICES.get(tool_choice, "auto")
        if self._responses or needs_responses(model):
            return self._tools_via_responses(client, messages, model, tools, opts, choice)
        try:
            resp = create_chat_completion(client, model, opts, messages=chat_messages(messages), tools=chat_tools(tools), tool_choice=choice)
        except NeedsResponsesAPI:
            return self._tools_via_responses(client, messages, model, tools, opts, choice)
        return chat_result(resp)

    @staticmethod
    def _tools_via_responses(client, messages, model, tools, opts, tool_choice) -> CompletionResult:
        return response_result(
            create_response(client, model, opts, input=response_input(messages), tools=response_tools(tools), tool_choice=tool_choice)
        )

    def complete_structured(self, messages: list[Message], model: str, output_schema: dict, options: CompletionOptions | None = None) -> dict:
        return self.execute_operation(messages, model, output_schema, options)[0]

    def execute_operation(self, messages, model, output_schema, options=None):
        opts = options or CompletionOptions()
        client = self._client()
        if self._responses:
            result = response_result(create_response(client, model, opts, input=response_input(messages), text=json_schema_format(output_schema)))
        else:
            result = chat_result(
                create_chat_completion(
                    client,
                    model,
                    opts,
                    messages=chat_messages(messages),
                    response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": output_schema, "strict": False}},
                )
            )
        return json.loads(result.content or "{}"), result

    def list_models(self) -> list[str]:
        models = self._client().models.list()
        return sorted(m.id for m in models.data if not any(marker in m.id for marker in NON_CHAT_MARKERS))

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        resp = self._client().embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    def test_connection(self) -> tuple[bool, str]:
        try:
            models = self.list_models()
            return True, f"Connected — {len(models)} models available"
        except Exception as e:  # noqa: BLE001 — reported to the admin, never raised
            return False, str(e)
