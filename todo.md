# ToDo:

- Full hexagonal architecture like already in place for ./scr/history
- Inclusion of proper database migrations using alembic https://alembic.sqlalchemy.org/en/latest/
- Config via https://docs.pydantic.dev/latest/api/config/
    - Configuration of the application should be possible via yaml.
    - In the yaml, environment keys should be injectable via `${env:KEY}`
- Human-In-The-Loop support for tool calls see https://ai.pydantic.dev/
- Full test-suite using pytest in ./tests

- Broader MCP support
    - Authentication with an encrypted access / refresh token store in the database https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization
    - Elicitation https://gofastmcp.com/servers/elicitation
    - MCP Apps https://modelcontextprotocol.io/extensions/apps/overview & https://gofastmcp.com/apps/overview

- Browser UI based on https://prefab.prefect.io/docs/welcome if possible.
    - MCP authentication should be integrated in the browser session then.
