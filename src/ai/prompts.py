from textwrap import dedent
from time import time_ns
from uuid import UUID, uuid4

from src.ai.models import SystemPrompt
from src.tools.models import ToolSet


class PromptsService:
    @staticmethod
    def _get_system_prompt_text_for_tool_sets(
        tool_sets: list[ToolSet],
        header_str: str | None = None,
    ) -> str:
        if header_str:
            header_str = header_str.strip()
        else:
            header_str = dedent("""
                # Tool Sets

                You have access to the following tool sets and tools:
            """).strip()

        system_prompt_texts: list[str] = [header_str]

        for tool_set in tool_sets:
            system_prompt_texts.append(
                dedent(f"""
                    ## {tool_set.name}

                    {tool_set.system_prompt}
                """).strip()
            )
            for tool in tool_set.tools:
                system_prompt_texts.append(
                    dedent(f"""
                        ### {tool.name}

                        {tool.system_prompt}
                    """).strip()
                )
        return "\n\n".join(system_prompt_texts)

    @staticmethod
    def _get_main_system_prompt(header_str: str | None = None) -> str:
        if header_str:
            return header_str.strip()
        else:
            return dedent("""
                [# General Instructions #]

                You are a helpful assistant .
            """).strip()

    @staticmethod
    def get_system_prompt(
        history_id: UUID,
        tool_sets: list[ToolSet],
        header_str: str | None = None,
        tool_section_header_str: str | None = None,
    ) -> SystemPrompt:
        main_system_prompt = PromptsService._get_main_system_prompt(header_str=header_str)
        tool_set_system_prompt = PromptsService._get_system_prompt_text_for_tool_sets(
            tool_sets=tool_sets,
            header_str=tool_section_header_str,
        )

        return SystemPrompt(
            id=uuid4(),
            history_id=history_id,
            created_at=time_ns(),
            prompt=dedent(f"""
                {main_system_prompt}
                {tool_set_system_prompt}
            """).strip(),
        )
