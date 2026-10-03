"""Minimal OpenAI-compatible HTTP transport.

Both target providers speak this shape - the DeepSeek API natively, and Ollama
through its ``/v1`` compatibility endpoint - so one small transport serves both.
That is why v0.1 needs no LiteLLM: the entire cost here is about sixty lines.

Uses ``urllib`` from the standard library, so the project has zero runtime
dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

from memory_core.domain.dto import ChatMessage, ChatResult
from memory_core.domain.errors import ProviderError
from memory_core.providers.base import to_wire_messages


def build_payload(model, messages, temperature=None, json_mode=False, extra=None):
    # type: (str, Sequence[ChatMessage], Optional[float], bool, Optional[Dict[str, Any]]) -> Dict[str, Any]
    payload = {
        "model": model,
        "messages": to_wire_messages(messages),
        "stream": False,
    }  # type: Dict[str, Any]
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if extra:
        payload.update(extra)
    return payload


def post_json(url, payload, headers=None, timeout=120.0):
    # type: (str, Dict[str, Any], Optional[Dict[str, str]], float) -> Dict[str, Any]
    """POST JSON and decode the response.

    Every transport failure becomes a :class:`ProviderError` carrying actionable
    context, because the caller's job is to degrade gracefully, not to debug HTTP.
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # pragma: no cover - best effort only
            detail = ""
        raise ProviderError("HTTP {0} from {1}: {2}".format(exc.code, url, detail))
    except urllib.error.URLError as exc:
        raise ProviderError("cannot reach {0}: {1}".format(url, exc.reason))
    except OSError as exc:
        raise ProviderError("transport error calling {0}: {1}".format(url, exc))
    try:
        data = json.loads(raw)
    except ValueError:
        raise ProviderError("provider returned non-JSON content from {0}".format(url))
    if not isinstance(data, dict):
        raise ProviderError("provider returned an unexpected payload from {0}".format(url))
    if "error" in data:
        raise ProviderError("provider error from {0}: {1}".format(url, data["error"]))
    return data


def extract_completion(data, provider, model):
    # type: (Dict[str, Any], str, str) -> ChatResult
    """Normalise an OpenAI-compatible completion into our own DTO."""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError("provider response contained no choices")
    first = choices[0] or {}
    message = first.get("message") or {}
    text = message.get("content")
    if text is None:
        text = first.get("text") or ""
    if not isinstance(text, str):
        text = str(text)
    usage = data.get("usage") or {}
    return ChatResult(
        text=text,
        provider=provider,
        model=str(data.get("model") or model),
        finish_reason=first.get("finish_reason"),
        usage=usage if isinstance(usage, dict) else {},
    )


def chat_completion(base_url, api_key, model, messages, timeout=120.0, temperature=None, json_mode=False, extra=None, provider="openai-compatible"):
    # type: (str, Optional[str], str, Sequence[ChatMessage], float, Optional[float], bool, Optional[Dict[str, Any]], str) -> ChatResult
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {}  # type: Dict[str, str]
    if api_key:
        headers["Authorization"] = "Bearer {0}".format(api_key)
    payload = build_payload(model, messages, temperature=temperature, json_mode=json_mode, extra=extra)
    data = post_json(url, payload, headers=headers, timeout=timeout)
    return extract_completion(data, provider, model)


__all__ = ["build_payload", "post_json", "extract_completion", "chat_completion", "List"]
