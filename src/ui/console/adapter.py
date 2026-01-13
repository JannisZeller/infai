from src.application.chat_use_case import ChatUseCase
from src.ui.console.service import ConsoleService
from src.ui.port import UI


class ConsoleAdapter(UI):
    """Driving adapter for the console interface."""

    def __init__(self, chat_use_case: ChatUseCase):
        self._chat_use_case = chat_use_case
        self._console_service = ConsoleService()

    async def run(self):
        """Run the console interaction loop."""
        while True:
            try:
                user_prompt_str = input('\n\n ❯ Enter your prompt ("q" to quit): ')
                if user_prompt_str.lower() == "q":
                    break
                if not user_prompt_str.strip():
                    continue

                stream = await self._chat_use_case.execute(user_prompt_str)
                await self._console_service.consume_stream(stream)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\nError: {e}")
