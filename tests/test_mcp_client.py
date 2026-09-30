"""Ai4ScienceMCPClient against an in-process MCP server that mimics the
ai4science server's community-tool tool. No network, no deployment."""

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel

from ai4science_client import (
    Ai4ScienceError,
    Ai4ScienceMCPClient,
    Ai4ScienceToolError,
    CommunityTool,
)

TOOLS = [
    {
        "domain": "chemistry",
        "name": "rdkit_descriptors",
        "version": "1.2.0",
        "title": "RDKit molecular descriptors",
        "description": "Compute molecular weight and logP for SMILES strings.",
        "authors": ["Jane Doe"],
        "license": "MIT",
        "input_schema": {"type": "object", "properties": {"smiles": {"type": "array"}}},
        "runtime": {
            "kind": "ephemeral",
            "entrypoint": "main.py",
            "dependencies": ["rdkit==2024.3.5"],
        },
        "resources": {"cpus": 1, "memory_mb": 2048},
        "a_field_from_a_newer_server": True,
    },
    {
        "domain": "astronomy",
        "name": "fits_header_summary",
        "version": "1.0.0",
        "title": "FITS header summary",
        "description": "Summarise the primary header of a FITS file.",
        "input_schema": {"type": "object"},
    },
]


class SearchResult(BaseModel):
    tools: list[dict]


def fake_server(archive_up: bool = True) -> MCPServer:
    server = MCPServer("hippocampus")

    @server.tool()
    def list_community_tools(
        domain: str | None = None, query: str | None = None
    ) -> SearchResult:
        """Search community tools."""
        if not archive_up:
            raise ToolError("Could not read the community tool archive.")
        needle = (query or "").lower()
        return SearchResult(
            tools=[
                t
                for t in TOOLS
                if (domain is None or t["domain"] == domain)
                and needle in f"{t['name']} {t['title']} {t['description']}".lower()
            ]
        )

    return server


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_list_tools_reports_the_servers_tools():
    async with Ai4ScienceMCPClient(server=fake_server()) as mcp:
        tools = await mcp.list_tools()
    assert [t.name for t in tools] == ["list_community_tools"]
    assert tools[0].input_schema["properties"].keys() == {"domain", "query"}


@pytest.mark.anyio
async def test_list_community_tools_returns_typed_models():
    async with Ai4ScienceMCPClient(server=fake_server()) as mcp:
        tools = await mcp.list_community_tools()
    assert all(isinstance(t, CommunityTool) for t in tools)
    rdkit = next(t for t in tools if t.name == "rdkit_descriptors")
    assert rdkit.qualified_name == "chemistry.rdkit_descriptors"
    assert rdkit.runtime.dependencies == ["rdkit==2024.3.5"]
    assert rdkit.resources.memory_mb == 2048


@pytest.mark.anyio
async def test_filters_are_passed_to_the_server():
    async with Ai4ScienceMCPClient(server=fake_server()) as mcp:
        by_domain = await mcp.list_community_tools(domain="astronomy")
        by_text = await mcp.list_community_tools(query="SMILES")
    assert [t.name for t in by_domain] == ["fits_header_summary"]
    assert [t.name for t in by_text] == ["rdkit_descriptors"]


@pytest.mark.anyio
async def test_get_community_tool():
    async with Ai4ScienceMCPClient(server=fake_server()) as mcp:
        tool = await mcp.get_community_tool("astronomy.fits_header_summary")
        assert tool.runtime is None  # optional fields tolerate older servers
        with pytest.raises(Ai4ScienceToolError, match="No published community tool"):
            await mcp.get_community_tool("astronomy.nothing")
        with pytest.raises(ValueError, match="<domain>.<name>"):
            await mcp.get_community_tool("no_dot")


@pytest.mark.anyio
async def test_tool_errors_become_ai4science_tool_error():
    async with Ai4ScienceMCPClient(server=fake_server(archive_up=False)) as mcp:
        with pytest.raises(Ai4ScienceToolError, match="community tool archive") as info:
            await mcp.list_community_tools()
    assert info.value.tool_name == "list_community_tools"


@pytest.mark.anyio
async def test_using_the_client_without_connecting_is_a_clear_error():
    mcp = Ai4ScienceMCPClient(server=fake_server())
    with pytest.raises(Ai4ScienceError, match="Not connected"):
        await mcp.list_tools()


def test_base_url_is_required_without_a_server(monkeypatch):
    monkeypatch.delenv("AI4SCIENCE_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="AI4SCIENCE_BASE_URL"):
        Ai4ScienceMCPClient()


def test_mcp_url_is_built_from_base_url():
    assert Ai4ScienceMCPClient("https://example.org/").url == "https://example.org/mcp"
