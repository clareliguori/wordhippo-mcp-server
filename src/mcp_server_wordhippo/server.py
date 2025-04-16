from typing import Annotated, Tuple
from urllib.parse import urlparse, urlunparse

import markdownify
import readabilipy.simple_json
from mcp.shared.exceptions import McpError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ErrorData,
    TextContent,
    Tool,
    INVALID_PARAMS,
    INTERNAL_ERROR,
)
from protego import Protego
from pydantic import BaseModel, Field

DEFAULT_USER_AGENT = "ModelContextProtocol/1.0 (Autonomous; +https://github.com/clareliguori/wordhippo-mcp-server)"

##############################################################################
# These methods are largely copied from the original fetch mcp server code.
# They deal with fetching a URL and converting the contents into markdown.
##############################################################################


def extract_content_from_html(html: str) -> str:
    """Extract and convert HTML content to Markdown format.

    Args:
        html: Raw HTML content to process

    Returns:
        Simplified markdown version of the content
    """
    ret = readabilipy.simple_json.simple_json_from_html_string(
        html, use_readability=True
    )
    if not ret["content"]:
        return "<error>Page failed to be simplified from HTML</error>"
    content = markdownify.markdownify(
        ret["content"],
        heading_style=markdownify.ATX,
    )
    return content


def get_robots_txt_url(url: str) -> str:
    """Get the robots.txt URL for a given website URL.

    Args:
        url: Website URL to get robots.txt for

    Returns:
        URL of the robots.txt file
    """
    # Parse the URL into components
    parsed = urlparse(url)

    # Reconstruct the base URL with just scheme, netloc, and /robots.txt path
    robots_url = urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))

    return robots_url


async def check_may_autonomously_fetch_url(
    url: str, user_agent: str, proxy_url: str | None = None
) -> None:
    """
    Check if the URL can be fetched by the user agent according to the robots.txt file.
    Raises a McpError if not.
    """
    from httpx import AsyncClient, HTTPError

    robot_txt_url = get_robots_txt_url(url)

    async with AsyncClient(proxies=proxy_url) as client:
        try:
            response = await client.get(
                robot_txt_url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
            )
        except HTTPError:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Failed to fetch robots.txt {robot_txt_url} due to a connection issue",
                )
            )
        if response.status_code in (401, 403):
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"When fetching robots.txt ({robot_txt_url}), received status {response.status_code} so assuming that autonomous fetching is not allowed.",
                )
            )
        elif 400 <= response.status_code < 500:
            return
        robot_txt = response.text
    processed_robot_txt = "\n".join(
        line for line in robot_txt.splitlines() if not line.strip().startswith("#")
    )
    robot_parser = Protego.parse(processed_robot_txt)
    if not robot_parser.can_fetch(str(url), user_agent):
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR,
                message=f"The sites robots.txt ({robot_txt_url}), specifies that autonomous fetching of this page is not allowed, "
                f"<useragent>{user_agent}</useragent>\n"
                f"<url>{url}</url>"
                f"<robots>\n{robot_txt}\n</robots>\n"
                f"The assistant must let the user know that it failed to view the page. The assistant may provide further guidance based on the above information.\n",
            )
        )


async def fetch_url(
    url: str, user_agent: str, proxy_url: str | None = None
) -> Tuple[str, str]:
    """
    Fetch the URL and return the content in a form ready for the LLM, as well as a prefix string with status information.
    """
    from httpx import AsyncClient, HTTPError

    async with AsyncClient(proxies=proxy_url) as client:
        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
                timeout=30,
            )
        except HTTPError as e:
            raise McpError(
                ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}")
            )
        if response.status_code >= 400:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Failed to fetch {url} - status code {response.status_code}",
                )
            )

        page_raw = response.text

    content_type = response.headers.get("content-type", "")
    is_page_html = (
        "<html" in page_raw[:100] or "text/html" in content_type or not content_type
    )

    if is_page_html:
        return extract_content_from_html(page_raw), ""

    return (
        page_raw,
        f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
    )


##############################################################################
# The core logic and definition of the WordHippo MCP server starts here.
##############################################################################


class WordHippoThesaurus(BaseModel):
    """Parameters for fetching similar words."""

    word: Annotated[
        str, Field(description="word that should be looked up in the thesaurus")
    ]


async def serve(
    custom_user_agent: str | None = None,
    ignore_robots_txt: bool = False,
    proxy_url: str | None = None,
) -> None:
    """Run the WordHippo MCP server.

    Args:
        custom_user_agent: Optional custom User-Agent string to use for requests
        ignore_robots_txt: Whether to ignore robots.txt restrictions
        proxy_url: Optional proxy URL to use for requests
    """
    server = Server("mcp-wordhippo")
    user_agent = custom_user_agent or DEFAULT_USER_AGENT

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="thesaurus",
                description="""Provides a list of similar words from a thesaurus.""",
                inputSchema=WordHippoThesaurus.model_json_schema(),
            )
        ]

    @server.call_tool()
    async def call_tool(name, arguments: dict) -> list[TextContent]:
        try:
            args = WordHippoThesaurus(**arguments)
        except ValueError as e:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))

        word = str(args.word)
        if not word:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Word is required"))

        url = f"https://www.wordhippo.com/what-is/another-word-for/{word}.html"

        if not ignore_robots_txt:
            await check_may_autonomously_fetch_url(url, user_agent, proxy_url)

        content, prefix = await fetch_url(url, user_agent, proxy_url=proxy_url)
        return [TextContent(type="text", text=f"{prefix}:\n{content}")]

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)
