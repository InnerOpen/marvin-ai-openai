"""Tool calling for OpenAI-shaped APIs: message translation and parsing (ported from Marvin core's
tests/test_provider_tool_calling.py). The SDK clients are mocked: no network, no key."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from marvin_integration_sdk.ai import CompletionResult, Message, ToolCall, ToolDefinition

from marvin_ai_openai import AzureOpenAIProvider, OpenAIProvider
from marvin_ai_openai.openai_api import chat_messages

TOOLS = [ToolDefinition(name="get_entry", description="Fetch an entry", input_schema={"type": "object"})]
COMPATIBLE = "http://localhost:8000/v1"  # an OpenAI-compatible server keeps Chat Completions


def _chat_response(content, calls, finish, usage=(10, 4)):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=sum(usage)),
        model="gpt-4o",
        model_dump=dict,
    )


def _with_client(provider, client):
    provider._client = MagicMock(return_value=client)
    return provider


def test_chat_messages_translate_a_tool_round_trip():
    api = chat_messages(
        [
            Message(role="system", content="You are Marvin."),
            Message(role="user", content="What's in the about page?"),
            Message(role="assistant", content="", tool_calls=[ToolCall(id="call_1", name="get_entry", arguments={"slug": "about"})]),
            Message(role="tool", content='{"title":"About"}', tool_call_id="call_1"),
        ]
    )
    assistant = api[2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] is None  # no text alongside the tool call
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_entry"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"slug": "about"}
    assert api[3] == {"role": "tool", "tool_call_id": "call_1", "content": '{"title":"About"}'}


def test_a_compatible_server_parses_tool_calls():
    tc = SimpleNamespace(id="call_9", type="function", function=SimpleNamespace(name="get_entry", arguments='{"slug":"home"}'))
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(None, [tc], "tool_calls")
    provider = _with_client(OpenAIProvider(api_key="x", base_url=COMPATIBLE), client)

    result = provider.complete_with_tools([Message(role="user", content="hi")], "gpt-4o", TOOLS)

    assert isinstance(result, CompletionResult)
    assert result.stop_reason == "tool_calls"
    assert result.tool_calls == [ToolCall(id="call_9", name="get_entry", arguments={"slug": "home"})]
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["tools"][0]["function"]["name"] == "get_entry"
    assert sent["tool_choice"] == "auto"


def test_a_compatible_server_plain_answer():
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response("Done.", None, "stop")
    provider = _with_client(OpenAIProvider(api_key="x", base_url=COMPATIBLE), client)

    result = provider.complete_with_tools([Message(role="user", content="hi")], "gpt-4o", TOOLS)
    assert result.content == "Done."
    assert result.tool_calls == []


def test_azure_parses_tool_calls():
    tc = SimpleNamespace(id="call_az", type="function", function=SimpleNamespace(name="get_entry", arguments='{"slug":"about"}'))
    client = MagicMock()
    client.chat.completions.create.return_value = _chat_response(None, [tc], "tool_calls", (8, 3))
    provider = _with_client(AzureOpenAIProvider(api_key="x", base_url="https://ex.openai.azure.com"), client)

    result = provider.complete_with_tools([Message(role="user", content="hi")], "gpt-4o", TOOLS)
    assert result.tool_calls == [ToolCall(id="call_az", name="get_entry", arguments={"slug": "about"})]
    assert client.chat.completions.create.call_args.kwargs["tools"][0]["function"]["name"] == "get_entry"


def test_tool_choice_required_reaches_the_responses_api():
    resp = SimpleNamespace(
        output=[],
        output_text="x",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        model="gpt-4o",
        incomplete_details=None,
        model_dump=dict,
    )
    client = MagicMock()
    client.responses.create.return_value = resp
    provider = _with_client(OpenAIProvider(api_key="x"), client)

    provider.complete_with_tools([Message(role="user", content="hi")], "gpt-4o", TOOLS, tool_choice="required")
    assert client.responses.create.call_args.kwargs["tool_choice"] == "required"
