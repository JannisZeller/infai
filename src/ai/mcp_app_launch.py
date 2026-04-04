import json
from typing import Any

from src.ai.models import MCPAppLaunchRequest
from src.history.models import ToolResult

_APP_MARKER_KEYS = {"type", "app", "launchUrl", "url", "resource_uri", "resourceUri", "_meta"}


def parse_mcp_app_launch_request(tool_result: ToolResult) -> MCPAppLaunchRequest | None:
    payload = _normalize_payload(tool_result.result)
    launch_payload = _find_launch_payload(payload)
    if launch_payload is None:
        return None

    url = _extract_launch_url(launch_payload)
    if url is None:
        return None

    return MCPAppLaunchRequest(
        id=tool_result.id,
        history_id=tool_result.history_id,
        created_at=tool_result.created_at,
        tool_call_id=tool_result.tool_call_id,
        tool_name=tool_result.tool_name,
        url=url,
        title=_extract_launch_title(launch_payload),
    )


def _normalize_payload(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload

    stripped = payload.strip()
    if not stripped.startswith(("{", "[")):
        return payload

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return payload


def _find_launch_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if _looks_like_launch_payload(payload):
            return payload

        nested_app = payload.get("app")
        if isinstance(nested_app, dict) and _looks_like_launch_payload(nested_app):
            combined = dict(nested_app)
            if "launchUrl" not in combined and isinstance(payload.get("launchUrl"), str):
                combined["launchUrl"] = payload["launchUrl"]
            return combined

        for value in payload.values():
            found = _find_launch_payload(value)
            if found is not None:
                return found

    if isinstance(payload, list):
        for item in payload:
            found = _find_launch_payload(item)
            if found is not None:
                return found

    return None


def _looks_like_launch_payload(payload: dict[str, Any]) -> bool:
    has_app_marker = bool(_APP_MARKER_KEYS & payload.keys())
    return has_app_marker and _extract_launch_url(payload) is not None


def _extract_launch_url(payload: dict[str, Any]) -> str | None:
    candidates = [
        payload.get("launchUrl"),
        payload.get("url"),
        payload.get("resource_uri"),
        payload.get("resourceUri"),
    ]

    app_payload = payload.get("app")
    if isinstance(app_payload, dict):
        candidates.extend(
            [
                app_payload.get("launchUrl"),
                app_payload.get("url"),
                app_payload.get("resource_uri"),
                app_payload.get("resourceUri"),
            ]
        )

    meta_payload = payload.get("_meta")
    if isinstance(meta_payload, dict):
        ui_payload = meta_payload.get("ui")
        if isinstance(ui_payload, dict):
            candidates.extend(
                [
                    ui_payload.get("launchUrl"),
                    ui_payload.get("url"),
                    ui_payload.get("resource_uri"),
                    ui_payload.get("resourceUri"),
                ]
            )

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            return candidate

    return None


def _extract_launch_title(payload: dict[str, Any]) -> str | None:
    candidates = [payload.get("title"), payload.get("name")]

    app_payload = payload.get("app")
    if isinstance(app_payload, dict):
        candidates.extend([app_payload.get("title"), app_payload.get("name")])

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate

    return None
