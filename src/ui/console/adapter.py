from src.application.chat_use_case import ChatUseCase
from src.core.logging import get_logger
from src.ui.console.service import ConsoleService
from src.ui.port import UI


class ConsoleAdapter(UI):
    """Driving adapter for the console interface."""

    def __init__(self, chat_use_case: ChatUseCase):
        self._chat_use_case = chat_use_case
        self._console_service = ConsoleService()
        self._logger = get_logger(__name__, output="console")

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
                approval_requests = await self._console_service.consume_stream(stream)

                while approval_requests:
                    resume_token = approval_requests[0].resume_token
                    approval_decisions = [
                        self._console_service.prompt_tool_approval(approval_request)
                        for approval_request in approval_requests
                    ]

                    stream = await self._chat_use_case.resume(
                        resume_token=resume_token,
                        approvals=approval_decisions,
                    )
                    approval_requests = await self._console_service.consume_stream(stream)
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Log with full traceback to help debug issues
                self._logger.error(f"Error processing request: {e}", exc_info=True)
