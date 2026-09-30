"""Discover community-contributed tools through the ai4science MCP endpoint.

The server exposes its tools over MCP at {base_url}/mcp -- the same
endpoint AI agents connect to. Among them is `list_community_tools`,
which searches the community tool archive: tools contributed by the
research community (chemistry, astronomy, ...), reviewed, and published.

This example plays the agent by hand:
  1. connect and see which tools the server offers,
  2. browse the community archive, by domain and by free-text search,
  3. inspect one community tool's inputs, runtime and resources.

Nothing here submits a job -- community tools are discoverable today;
running them comes in a later release.

Run:
    AI4SCIENCE_BASE_URL=https://ai4science.dev.sdp.surf.nl \\
        uv run python examples/mcp_community_tools.py

(or put AI4SCIENCE_BASE_URL in your .env; add AI4SCIENCE_API_KEY if the
server requires one)
"""

import asyncio
import json

from ai4science_client import (
    Ai4ScienceAPIError,
    Ai4ScienceConnectionError,
    Ai4ScienceMCPClient,
    Ai4ScienceToolError,
)


async def main() -> None:
    async with Ai4ScienceMCPClient() as mcp:
        # 1. What does the server offer agents?
        print("Tools on the MCP endpoint:")
        for tool in await mcp.list_tools():
            kind = "read-only" if tool.read_only else "acts"
            print(f"  - {tool.name} ({kind}): {tool.title}")

        # 2. Browse the community tool archive.
        try:
            community = await mcp.list_community_tools()
        except Ai4ScienceToolError as e:
            print(f"\nCommunity archive unavailable: {e}")
            return

        print(f"\nCommunity tools ({len(community)} published):")
        for tool in community:
            print(f"  - {tool.qualified_name} v{tool.version}: {tool.title}")
        if not community:
            print(
                "  (none yet -- submit one via POST /api/v1/community-tools/submissions)"
            )
            return

        domains = sorted({tool.domain for tool in community})
        print(f"\nDomains: {', '.join(domains)}")
        for domain in domains:
            in_domain = await mcp.list_community_tools(domain=domain)
            print(f"  {domain}: {[tool.name for tool in in_domain]}")

        hits = await mcp.list_community_tools(query="molecul")
        print(f"\nFree-text search 'molecul': {[t.qualified_name for t in hits]}")

        # 3. Inspect one tool: what it takes, how it runs, what it needs.
        tool = await mcp.get_community_tool(community[0].qualified_name)
        print(f"\n{tool.qualified_name} v{tool.version} by {', '.join(tool.authors)}")
        print(f"  {tool.description}")
        print(f"  license:   {tool.license}")
        if tool.runtime:
            deps = tool.runtime.dependencies or tool.runtime.modules
            print(
                f"  runtime:   {tool.runtime.kind}, entrypoint {tool.runtime.entrypoint}, {deps}"
            )
        if tool.resources:
            r = tool.resources
            gpu = ", GPU" if r.needs_gpu else ""
            print(
                f"  resources: {r.cpus} CPU, {r.memory_mb} MB, {r.time_limit_minutes} min{gpu}"
            )
        print("  input schema:")
        print("    " + json.dumps(tool.input_schema, indent=2).replace("\n", "\n    "))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Ai4ScienceConnectionError as e:
        raise SystemExit(f"Could not reach the server: {e}") from None
    except Ai4ScienceAPIError as e:
        raise SystemExit(f"The server refused: {e}") from None
