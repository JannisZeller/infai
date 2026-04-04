import json
from collections.abc import Mapping
from typing import Any, cast

from cryptography.fernet import Fernet, InvalidToken


class TokenStoreCipher:
    def __init__(self, encryption_key: str):
        try:
            self._fernet = Fernet(encryption_key.encode("utf-8"))
        except ValueError as exc:
            raise ValueError("Token store encryption key must be a valid Fernet key.") from exc

    def encrypt(self, value: Mapping[str, Any]) -> bytes:
        normalized = dict(value)
        if not all(isinstance(key, str) for key in normalized):
            raise ValueError("Token store values must be JSON objects with string keys.")

        plaintext = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self._fernet.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes) -> dict[str, Any]:
        try:
            plaintext = self._fernet.decrypt(ciphertext)
        except InvalidToken as exc:
            raise ValueError("Unable to decrypt token-store value with configured encryption key.") from exc

        loaded = json.loads(plaintext)
        if not isinstance(loaded, dict):
            raise ValueError("Token store payload must deserialize to a JSON object.")

        return cast(dict[str, Any], loaded)
