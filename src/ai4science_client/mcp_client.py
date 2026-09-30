"""Client for the ai4science agentic layer: the MCP endpoint at {base_url}/mcp.

This is the same endpoint AI agents connect to. Using it from Python lets
you discover and call the server's tools directly -- including searching
the community tool archive (tools contributed by the research community,
e.g. for chemistry or astronomy).

Async by design, like the MCP protocol itself::

    async with Ai4ScienceMCPClient(base_url) as mcp:
        tools = await mcp.list_community_tools(domain="chemistry")

In a Jupyter notebook, `await` works at the top level of a cell. In a
script, wrap your code in `asyncio.run(...)`.

Requires the `mcp` package (imported lazily, so the rest of
ai4science_client works without it).
"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack
from typing import Any

from .exceptions import (
    Ai4ScienceAPIError,
    Ai4ScienceConnectionError,
    Ai4ScienceError,
    Ai4ScienceToolError,
)
from .schemas import CommunityTool, McpTool

MCP_PATH = "/mcp"


class Ai4ScienceMCPClient:
    """Async client for an ai4science deployment's MCP endpoint.

    Parameters
    ----------
    base_url : str | None
        e.g. "https://ai4science.dev.sdp.surf.nl". Falls back to
        AI4SCIENCE_BASE_URL if not given.
    api_key : str | None
        Sent as X-API-Key, only needed if the server sets API_KEYS.
        Falls back to AI4SCIENCE_API_KEY if not given.
    server : Any | None
        Advanced/testing: an in-process MCP server object to connect to
        instead of base_url (anything `mcp.Client` accepts).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        server: Any | None = None,
    ):
        if server is None:
            from . import config as _config  # noqa: F401, PLC0415  (loads .env)

            base_url = base_url or os.environ.get("AI4SCIENCE_BASE_URL")
            if not base_url:
                raise ValueError(
                    "Missing base_url. Pass it explicitly or set AI4SCIENCE_BASE_URL."
                )
        self.url = f"{base_url.rstrip('/')}{MCP_PATH}" if base_url else None
        self.api_key = api_key or os.environ.get("AI4SCIENCE_API_KEY") or None
        self._server = server
        self._stack: AsyncExitStack | None = None
        self._session: Any = None

    # --- connection lifecycle ---

    async def __aenter__(self) -> Ai4ScienceMCPClient:
        try:
            from mcp import Client  # noqa: PLC0415
            from mcp.client.streamable_http import (  # noqa: PLC0415
                create_mcp_http_client,
                streamable_http_client,
            )
        except ImportError as e:
            raise ImportError(
                "Ai4ScienceMCPClient needs the 'mcp' package: pip install 'mcp>=2.2,<3'"
            ) from e

        stack = AsyncExitStack()
        rejected: list[tuple[int, str | None]] = []
        try:
            if self._server is not None:
                self._session = await stack.enter_async_context(Client(self._server))
            else:
                headers = {"X-API-Key": self.api_key} if self.api_key else {}
                http = await stack.enter_async_context(
                    create_mcp_http_client(headers=headers)
                )
                # The MCP SDK hides HTTP error statuses behind a generic
                # protocol error; record them so we can say *why*.
                http.event_hooks["response"].append(_recorder(rejected))
                self._session = await stack.enter_async_context(
                    Client(streamable_http_client(self.url, http_client=http))
                )
        except Exception as e:
            await stack.aclose()
            if rejected:
                status, detail = rejected[0]
                raise Ai4ScienceAPIError(
                    f"MCP endpoint {self.url} rejected the connection: {detail or 'no detail'}",
                    status_code=status,
                    detail=detail,
                ) from e
            raise Ai4ScienceConnectionError(
                f"Could not connect to the MCP endpoint at {self.url or 'in-process server'}: "
                f"{_root_cause(e)}"
            ) from e
        self._stack = stack
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None

    # --- generic MCP ---

    async def list_tools(self) -> list[McpTool]:
        """Every tool the server offers to agents."""
        result = await self._require_session().list_tools()
        return [
            McpTool(
                name=tool.name,
                title=tool.title,
                description=" ".join((tool.description or "").split()),
                read_only=bool(tool.annotations and tool.annotations.read_only_hint),
                input_schema=tool.input_schema or {},
                output_schema=tool.output_schema,
            )
            for tool in result.tools
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> Any:
        """Call any server tool by name. Returns its structured result.

        Raises
        ------
        Ai4ScienceToolError
            If the tool reports an error (the message says why).
        """
        result = await self._require_session().call_tool(name, arguments or {})
        if result.is_error:
            raise Ai4ScienceToolError(_text_of(result), tool_name=name)
        if result.structured_content is not None:
            return result.structured_content
        return _text_of(result)

    # --- community tools ---

    async def list_community_tools(
        self, domain: str | None = None, query: str | None = None
    ) -> list[CommunityTool]:
        """Search the community tool archive: latest version of each
        published tool, optionally filtered by domain (e.g. "chemistry")
        and/or free text matched against name, title and description."""
        arguments = {k: v for k, v in {"domain": domain, "query": query}.items() if v}
        result = await self.call_tool("list_community_tools", arguments)
        return [CommunityTool.model_validate(tool) for tool in result["tools"]]

    async def get_community_tool(self, qualified_name: str) -> CommunityTool:
        """One community tool by its '<domain>.<name>' name, e.g.
        'chemistry.rdkit_descriptors'."""
        domain, _, name = qualified_name.partition(".")
        if not domain or not name:
            raise ValueError(
                f"Expected '<domain>.<name>', e.g. 'chemistry.rdkit_descriptors', "
                f"got {qualified_name!r}."
            )
        for tool in await self.list_community_tools(domain=domain):
            if tool.name == name:
                return tool
        raise Ai4ScienceToolError(
            f"No published community tool named {qualified_name!r}.",
            tool_name="list_community_tools",
        )

    # --- internals ---

    def _require_session(self) -> Any:
        if self._session is None:
            raise Ai4ScienceError(
                "Not connected. Use `async with Ai4ScienceMCPClient(...) as mcp:`."
            )
        return self._session


def _text_of(result: Any) -> str:
    return "\n".join(
        block.text for block in result.content if getattr(block, "text", None)
    )


def _recorder(rejected: list[tuple[int, str | None]]) -> Any:
    """httpx response hook: remember the first HTTP error response."""

    async def record(response: Any) -> None:
        if response.status_code >= 400 and not rejected:
            await response.aread()
            detail: str | None
            try:
                body = response.json()
                detail = body.get("detail") if isinstance(body, dict) else None
            except ValueError:
                detail = response.text or None
            rejected.append((response.status_code, detail))

    return record


def _root_cause(error: BaseException) -> str:
    """Unwrap anyio ExceptionGroups to the first real error message."""
    # Duck-typed rather than isinstance(..., BaseExceptionGroup), which
    # only exists on Python 3.11+.
    while getattr(error, "exceptions", None):
        error = error.exceptions[0]  # type: ignore[attr-defined]
    return f"{type(error).__name__}: {error}"
