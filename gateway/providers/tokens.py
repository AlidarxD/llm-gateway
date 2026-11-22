"""Cheap token estimate (~4 chars/token) used when a provider reports no usage."""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0
