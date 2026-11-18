from __future__ import annotations

import json
from typing import Any, Callable

import httpx2
import pytest

from openai import OpenAI, AsyncOpenAI, APIResponseValidationError

# Clearly fake. Never a real credential.
_API_KEY = "sk-test-json-content-type"

# RFC 9110: type/subtype is case-insensitive, and optional whitespace may appear
# before a parameter semicolon. Parameter values themselves are not normalized.
_JSON_CONTENT_TYPES = [
    "application/json",
    "Application/JSON",
    "application/json ; charset=utf-8",
    "Application/JSON ; charset=UTF-8",
    "application/json\t;charset=utf-8",
    "application/vnd.api+JSON",
]

_OBJECT = {"zero": 0, "one": 1}
_LIST = [0, "one"]


class _Transport(httpx2.BaseTransport, httpx2.AsyncBaseTransport):
    def __init__(self, handler: Callable[[httpx2.Request], httpx2.Response]) -> None:
        self._handler = handler

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        return self._handler(request)

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return self._handler(request)


def _json_response(content_type: str, payload: object) -> httpx2.Response:
    return httpx2.Response(
        200,
        content=json.dumps(payload).encode(),
        headers={"Content-Type": content_type},
    )


def _sync_client(handler: Callable[[httpx2.Request], httpx2.Response], *, strict: bool) -> OpenAI:
    return OpenAI(
        api_key=_API_KEY,
        _strict_response_validation=strict,
        http_client=httpx2.Client(transport=_Transport(handler)),
    )


def _async_client(handler: Callable[[httpx2.Request], httpx2.Response], *, strict: bool) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_API_KEY,
        _strict_response_validation=strict,
        http_client=httpx2.AsyncClient(transport=_Transport(handler)),
    )


def _parse_sync(client: OpenAI, *, mode: str, cast_to: type[Any], content_type: str) -> Any:
    if mode == "ordinary":
        return client.get("/probe", cast_to=cast_to)
    if mode == "raw":
        # Public raw responses still go through LegacyAPIResponse.
        response = client.models.with_raw_response.retrieve("gpt-4")
        assert response.headers["content-type"] == content_type
        return response.parse(to=cast_to)
    if mode == "streaming":
        with client.models.with_streaming_response.retrieve("gpt-4") as response:
            assert response.headers["content-type"] == content_type
            return response.parse(to=cast_to)
    raise AssertionError(mode)


async def _parse_async(client: AsyncOpenAI, *, mode: str, cast_to: type[Any], content_type: str) -> Any:
    if mode == "ordinary":
        return await client.get("/probe", cast_to=cast_to)
    if mode == "raw":
        response = await client.models.with_raw_response.retrieve("gpt-4")
        assert response.headers["content-type"] == content_type
        return response.parse(to=cast_to)
    if mode == "streaming":
        async with client.models.with_streaming_response.retrieve("gpt-4") as response:
            assert response.headers["content-type"] == content_type
            return await response.parse(to=cast_to)
    raise AssertionError(mode)


@pytest.mark.parametrize("content_type", _JSON_CONTENT_TYPES)
@pytest.mark.parametrize("strict", [False, True], ids=["loose", "strict"])
@pytest.mark.parametrize("mode", ["ordinary", "raw", "streaming"])
@pytest.mark.parametrize("cast_to,payload", [(dict, _OBJECT), (list, _LIST)], ids=["dict", "list"])
def test_sync_json_content_type_variants(
    content_type: str,
    strict: bool,
    mode: str,
    cast_to: type[Any],
    payload: object,
) -> None:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return _json_response(content_type, payload)

    with _sync_client(respond, strict=strict) as client:
        value = _parse_sync(client, mode=mode, cast_to=cast_to, content_type=content_type)

    assert type(value) is cast_to
    assert value == payload


@pytest.mark.parametrize("content_type", _JSON_CONTENT_TYPES)
@pytest.mark.parametrize("strict", [False, True], ids=["loose", "strict"])
@pytest.mark.parametrize("mode", ["ordinary", "raw", "streaming"])
@pytest.mark.parametrize("cast_to,payload", [(dict, _OBJECT), (list, _LIST)], ids=["dict", "list"])
async def test_async_json_content_type_variants(
    content_type: str,
    strict: bool,
    mode: str,
    cast_to: type[Any],
    payload: object,
) -> None:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return _json_response(content_type, payload)

    async with _async_client(respond, strict=strict) as client:
        value = await _parse_async(client, mode=mode, cast_to=cast_to, content_type=content_type)

    assert type(value) is cast_to
    assert value == payload


@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "text/plain ; charset=utf-8", " Application/Text ; charset=utf-8"],
)
@pytest.mark.parametrize("mode", ["ordinary", "raw"])
def test_sync_non_json_content_type_is_unchanged(content_type: str, mode: str) -> None:
    body = '{"zero": 0}'

    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body.encode(), headers={"Content-Type": content_type})

    with _sync_client(respond, strict=False) as client:
        value = _parse_sync(client, mode=mode, cast_to=dict, content_type=content_type)
    assert value == body

    media_type, *_ = content_type.split(";")
    with _sync_client(respond, strict=True) as client:
        with pytest.raises(APIResponseValidationError, match="Expected Content-Type") as exc:
            _parse_sync(client, mode=mode, cast_to=dict, content_type=content_type)
    assert f"received `{media_type}`" in str(exc.value)


@pytest.mark.parametrize("mode", ["ordinary", "raw"])
def test_sync_malformed_json_on_whitespace_content_type(mode: str) -> None:
    def respond(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=b"{", headers={"Content-Type": "application/json ; charset=utf-8"})

    with _sync_client(respond, strict=False) as client:
        with pytest.raises(json.JSONDecodeError):
            _parse_sync(client, mode=mode, cast_to=dict, content_type="application/json ; charset=utf-8")
