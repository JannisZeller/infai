import json
from pathlib import Path

from src.history.models import History, HistoryItem


class HistoryService:
    def __init__(self, path: Path):
        self.path = path
        self._history = None

    def _load_history(self) -> History:
        self._create_path_if_not_exists()
        with open(self.path, "r") as f:
            data = f.read()
            if not data:
                return History(items=[])
            return History.model_validate_json(data)

    def _get_history(self) -> History:
        if self._history is None:
            self._history = self._load_history()
        return self._history

    @property
    def history(self) -> History:
        return self._get_history()

    def _create_path_if_not_exists(self):
        if not self.path.exists():
            self.path.parent.mkdir(parents=True)
            self.path.touch()

    def save(self):
        self._create_path_if_not_exists()
        # Serialize each item individually to ensure all subclass fields are included
        items_data = [item.model_dump(mode="json") for item in self.history.items]
        history_data = {"items": items_data}
        with open(self.path, "w") as f:
            json.dump(history_data, f, indent=4)

    def add_history_items(self, history_items: list[HistoryItem]):
        self._create_path_if_not_exists()
        history = self.history
        history.items.extend(history_items)
        self.save()
