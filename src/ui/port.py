from typing import Protocol


class UI(Protocol):
    """The Port that the UI expects."""

    async def run(self):
        pass
