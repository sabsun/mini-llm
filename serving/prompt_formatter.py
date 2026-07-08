from typing import List

from serving.schemas import ChatMessage


class PromptFormatter:
    """
    Converts chat messages into a prompt string.

    This is intentionally separated from the API layer so that
    prompt formatting can evolve independently.
    """

    @staticmethod
    def format(messages: List[ChatMessage]) -> str:
        lines = []

        for message in messages:
            role = message.role.lower()

            if role == "system":
                lines.append(f"System: {message.content}")

            elif role == "user":
                lines.append(f"User: {message.content}")

            elif role == "assistant":
                lines.append(f"Assistant: {message.content}")

        lines.append("Assistant:")

        return "\n".join(lines)