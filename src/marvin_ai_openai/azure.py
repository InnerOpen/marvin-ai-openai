"""Azure OpenAI: the same models behind an Azure resource, through Chat Completions.

It shares openai_api with the OpenAI provider (request parameters learned from a 400, the message
shapes). A deployment that refuses tools while it reasons moves that deployment's tool calls to the
Responses API, which needs an ``api_version`` that has it.
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
    needs_responses,
    response_input,
    response_result,
    response_tools,
)
from .prices import AZURE_PRICES
from .provider import openai_client

DEFAULT_API_VERSION = "2024-02-01"


class AzureOpenAIProvider(AIProvider):
    provider_type = "azure"
    display_name = "Azure OpenAI"
    credentials = (
        API_KEY,
        Credential("base_url", "Endpoint", required=True, help="https://<resource>.openai.azure.com"),
        Credential("api_version", "API version", default=DEFAULT_API_VERSION),
    )
    supports_vision = True
    supports_structured_output = True
    supports_embeddings = True
    supports_tool_calls = True
    prices = AZURE_PRICES
    default_embedding_model = "text-embedding-3-small"

    def __init__(self, api_key: str, base_url: str, api_version: str = DEFAULT_API_VERSION, **client_options: Any) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._api_version = api_version
        self._client_options = client_options

    @classmethod
    def from_credentials(cls, values: Mapping[str, Any]) -> AzureOpenAIProvider:
        return cls(
            api_key=values.get("api_key") or "", base_url=values.get("base_url") or "", api_version=values.get("api_version") or DEFAULT_API_VERSION
        )

    def _client(self):
        return openai_client(
            "AzureOpenAI", api_key=self._api_key, azure_endpoint=self._base_url, api_version=self._api_version, **self._client_options
        )

    def complete(self, messages: list[Message], model: str, options: CompletionOptions | None = None) -> CompletionResult:
        return chat_result(create_chat_completion(self._client(), model, options or CompletionOptions(), messages=chat_messages(messages)))

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
        try:
            if needs_responses(model):
                raise NeedsResponsesAPI(model)
            return chat_result(
                create_chat_completion(client, model, opts, messages=chat_messages(messages), tools=chat_tools(tools), tool_choice=choice)
            )
        except NeedsResponsesAPI:
            return response_result(
                create_response(client, model, opts, input=response_input(messages), tools=response_tools(tools), tool_choice=choice)
            )

    def complete_structured(self, messages: list[Message], model: str, output_schema: dict, options: CompletionOptions | None = None) -> dict:
        resp = create_chat_completion(
            self._client(),
            model,
            options or CompletionOptions(),
            messages=chat_messages(messages),
            response_format={"type": "json_schema", "json_schema": {"name": "output", "schema": output_schema}},
        )
        return json.loads(resp.choices[0].message.content or "{}")

    # execute_operation: the SDK default (complete, then parse the JSON answer). JSON-schema output needs an
    # api_version newer than the default, so operations don't depend on it.

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self._client().models.list().data]
        except Exception:  # noqa: BLE001 — an Azure key may not list models; deployments still work
            return []

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        resp = self._client().embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    def test_connection(self) -> tuple[bool, str]:
        try:
            self._client().models.list()
            return True, "Connected to Azure OpenAI"
        except Exception as e:  # noqa: BLE001 — reported to the admin, never raised
            return False, str(e)
