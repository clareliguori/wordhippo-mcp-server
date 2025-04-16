# wordhippo-mcp-server

Thesaurus MCP server

Fork of the [mcp fetch](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch) server and modified to only fetch from WordHippo, given a word

### Usage

```json
"mcpServers": {
  "thesaurus": {
    "command": "docker",
    "args": ["run", "-i", "--rm", "ghcr.io/clareliguori/wordhippo-mcp-server"]
  }
}
```

### Development

```bash
uv venv
source .venv/bin/activate

uv sync --all-extras --dev

docker build -t ghcr.io/clareliguori/wordhippo-mcp-server:latest .

docker push ghcr.io/clareliguori/wordhippo-mcp-server:latest
```
