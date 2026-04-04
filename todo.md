# ToDo:

- Phase 1: Foundation ✅
    - Database migrations via Alembic ✅
    - Pydantic config via YAML with `${env:KEY}` interpolation ✅
    - Human-In-The-Loop support for tool calls ✅
    - Full test suite via pytest ✅
    - Config refactor from database `url` to `connection_string` ✅
    - Tool approval refactor so `requires_approval` only exists on `Tool` ✅

- Phase 2: MCP filtering and elicitation
    - MCP servers should be filterable, i.e., tools must be whitelisted.
    - Elicitation support https://gofastmcp.com/servers/elicitation

- Phase 3: MCP auth without browser UI
    - Authentication with an encrypted access / refresh token store in the database https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization
    - The console UI version opens a browser only for the OAuth flow.

- Phase 4: MCP apps without browser UI
    - MCP Apps https://modelcontextprotocol.io/extensions/apps/overview & https://gofastmcp.com/apps/overview
    - The console UI version opens a browser only for the MCP app.

- Phase 5: Browser UI
    - Browser UI based on https://prefab.prefect.io/docs/welcome if possible.
    - If prefab is not sufficient for what we want to achieve: Restructure the repo with the current python project being the "backend" (or similar as the console is a UI but included there)
      and a simple frontend (vite + React). For the frontend use shadcn https://ui.shadcn.com/
    - MCP authentication should be integrated in the browser session then.
    - MCP apps should be directly integrated in the browser UI.
