from collections.abc import Callable
from typing import Any

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext

ElicitContentValue = str | int | float | bool | list[str] | None


class ConsoleMCPElicitationService:
    def __init__(
        self,
        prompt_user: Callable[[str], str] = input,
        print_line: Callable[[str], None] = print,
    ):
        self._prompt_user = prompt_user
        self._print_line = print_line

    async def handle_elicitation(
        self,
        _context: RequestContext[ClientSession, Any, Any],
        params: mcp_types.ElicitRequestParams,
    ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
        if isinstance(params, mcp_types.ElicitRequestURLParams):
            return self._handle_url_elicitation(params)
        return self._handle_form_elicitation(params)

    def _handle_url_elicitation(self, params: mcp_types.ElicitRequestURLParams) -> mcp_types.ElicitResult:
        self._print_line("\n🔐 MCP elicitation required (URL)")
        self._print_line(params.message)
        self._print_line(f"Open this URL manually to continue: {params.url}")

        while True:
            response = self._prompt_user("Mark this elicitation as completed? [y/N/c]: ").strip().lower()
            if response in {"y", "yes"}:
                return mcp_types.ElicitResult(action="accept")
            if response in {"n", "no", ""}:
                return mcp_types.ElicitResult(action="decline")
            if response in {"c", "cancel"}:
                return mcp_types.ElicitResult(action="cancel")

    def _handle_form_elicitation(self, params: mcp_types.ElicitRequestFormParams) -> mcp_types.ElicitResult:
        self._print_line("\n📝 MCP elicitation required (form)")
        self._print_line(params.message)

        schema = params.requestedSchema
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(properties, dict):
            return mcp_types.ElicitResult(action="accept", content={})

        required_raw = schema.get("required") if isinstance(schema, dict) else None
        required_fields = set(required_raw) if isinstance(required_raw, list) else set()

        content: dict[str, ElicitContentValue] = {}
        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, dict):
                continue

            is_required = field_name in required_fields
            value = self._prompt_field_value(
                field_name=field_name,
                field_schema=field_schema,
                is_required=is_required,
            )
            content[field_name] = value

        decision = self._prompt_user("Submit elicitation response? [Y/n/c]: ").strip().lower()
        if decision in {"n", "no"}:
            return mcp_types.ElicitResult(action="decline")
        if decision in {"c", "cancel"}:
            return mcp_types.ElicitResult(action="cancel")
        return mcp_types.ElicitResult(action="accept", content=content)

    def _prompt_field_value(
        self,
        field_name: str,
        field_schema: dict[str, Any],
        is_required: bool,
    ) -> ElicitContentValue:
        schema_type_raw = field_schema.get("type")
        schema_type = schema_type_raw if isinstance(schema_type_raw, str) else "string"
        enum_values_raw = field_schema.get("enum")
        enum_values = enum_values_raw if isinstance(enum_values_raw, list) else None

        required_label = "required" if is_required else "optional"
        schema_label = schema_type
        if enum_values:
            schema_label += f", one of: {', '.join(str(v) for v in enum_values)}"

        while True:
            raw_value = self._prompt_user(f"{field_name} ({required_label}, {schema_label}): ").strip()
            if not raw_value:
                if is_required:
                    self._print_line("A value is required.")
                    continue
                return None

            try:
                parsed_value = self._parse_field_value(raw_value=raw_value, schema_type=schema_type)
            except ValueError as exc:
                self._print_line(str(exc))
                continue

            if enum_values is not None and parsed_value not in enum_values:
                self._print_line(f"Please pick one of: {', '.join(str(v) for v in enum_values)}")
                continue

            return parsed_value

    @staticmethod
    def _parse_field_value(raw_value: str, schema_type: str) -> ElicitContentValue:
        if schema_type == "integer":
            return int(raw_value)
        if schema_type == "number":
            return float(raw_value)
        if schema_type == "boolean":
            lowered = raw_value.lower()
            if lowered in {"true", "t", "1", "yes", "y"}:
                return True
            if lowered in {"false", "f", "0", "no", "n"}:
                return False
            raise ValueError("Enter a boolean value (yes/no, true/false).")
        if schema_type == "array":
            return [part.strip() for part in raw_value.split(",") if part.strip()]
        return raw_value
