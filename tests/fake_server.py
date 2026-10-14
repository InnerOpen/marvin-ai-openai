"""A fake OpenAI (and Azure OpenAI) server for the conformance kit: the real ``openai`` SDK talks to it
through ``httpx.MockTransport``, so requests and answers go through the SDK's own encoding and parsing."""

from __future__ import annotations

import json
from collections import deque

import httpx
from marvin_integration_sdk.ai import ToolCall
from marvin_integration_sdk.ai.fake import FakeTransport


class FakeOpenAIServer(FakeTransport):
    """Queued neutral replies, answered in the wire shape of whichever endpoint the provider calls."""

    def __init__(self) -> None:
        self.replies: deque[dict] = deque()
        self.requests: list[httpx.Request] = []

    # ── the kit's vocabulary ─────────────────────────────────────────────────

    def reply_text(self, text: str, *, prompt_tokens: int = 11, completion_tokens: int = 7, cached: int = 0) -> None:
        """`cached`: the part of the prompt OpenAI's automatic cache served, reported inside the prompt count."""
        self.replies.append({"text": text, "usage": (prompt_tokens, completion_tokens), "cached": cached})

    def reply_tool_calls(self, calls: list[ToolCall], *, prompt_tokens: int = 11, completion_tokens: int = 7) -> None:
        self.replies.append({"tool_calls": list(calls), "usage": (prompt_tokens, completion_tokens)})

    def reply_models(self, ids: list[str]) -> None:
        self.replies.append({"models": list(ids)})

    def reply_embeddings(self, vectors: list[list[float]]) -> None:
        self.replies.append({"embeddings": [list(v) for v in vectors]})

    def reply_error(self, status: int, message: str) -> None:
        self.replies.append({"error": (status, message)})

    def last_request(self) -> str:
        return self.requests[-1].content.decode() if self.requests else ""

    def last_json(self) -> dict:
        return json.loads(self.last_request() or "{}")

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    # ── the server ───────────────────────────────────────────────────────────

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.popleft() if self.replies else {}
        if "error" in reply:
            status, message = reply["error"]
            return httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error", "param": None, "code": None}})
        path = request.url.path
        model = json.loads(request.content or b"{}").get("model", "") if request.method == "POST" else ""
        if path.endswith("/responses"):
            return httpx.Response(200, json=_response(reply, model))
        if path.endswith("/chat/completions"):
            return httpx.Response(200, json=_chat(reply, model or path.split("/deployments/")[-1].split("/")[0]))
        if path.endswith("/embeddings"):
            data = [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(reply.get("embeddings", []))]
            return httpx.Response(200, json={"object": "list", "data": data, "model": model, "usage": {"prompt_tokens": 1, "total_tokens": 1}})
        if path.endswith("/models"):
            data = [{"id": i, "object": "model", "created": 0, "owned_by": "fake"} for i in reply.get("models", [])]
            return httpx.Response(200, json={"object": "list", "data": data})
        return httpx.Response(404, json={"error": {"message": f"no fake for {path}", "type": "not_found"}})


def _usage(reply: dict) -> tuple[int, int]:
    return reply.get("usage", (0, 0))


def _response(reply: dict, model: str) -> dict:
    pt, ct = _usage(reply)
    output: list[dict] = [
        {"type": "function_call", "id": f"fc_{c.id}", "call_id": c.id, "name": c.name, "arguments": json.dumps(c.arguments), "status": "completed"}
        for c in reply.get("tool_calls", [])
    ]
    if "text" in reply:
        output.append(
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": reply["text"], "annotations": []}],
            }
        )
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": model,
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "incomplete_details": None,
        "usage": {
            "input_tokens": pt,
            "output_tokens": ct,
            "total_tokens": pt + ct,
            "input_tokens_details": {"cached_tokens": reply.get("cached", 0)},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _chat(reply: dict, model: str) -> dict:
    pt, ct = _usage(reply)
    calls = [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in reply.get("tool_calls", [])
    ]
    message = {"role": "assistant", "content": reply.get("text"), **({"tool_calls": calls} if calls else {})}
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message}],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "prompt_tokens_details": {"cached_tokens": reply.get("cached", 0)},
        },
    }
