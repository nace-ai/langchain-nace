"""Prepare agent messages for Drex inside middleware.

Middleware must not stop an agent because a conversation holds media the decision
model can't read, so here media becomes text placeholders instead of an error (the
classifier itself refuses rather than dropping anything).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

_IMAGE_TYPES = frozenset({"image", "image_url"})


def _is_inline_image(block: dict[str, Any]) -> bool:
    if block.get("type") not in _IMAGE_TYPES:
        return False
    if block.get("source_type") == "base64" or isinstance(block.get("base64"), str):
        return True
    image_url = block.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    url = url if isinstance(url, str) else block.get("url")
    return isinstance(url, str) and url.startswith("data:")


def without_media(message: BaseMessage, *, keep_images: bool) -> BaseMessage:
    """Copy `message` with media replaced by "[... omitted]" text.

    Inline images are kept when `keep_images` is set; remote image URLs, audio,
    video and files are always replaced.
    """
    if isinstance(message.content, str):
        return message
    content: list[Any] = []
    for block in message.content:
        if (
            isinstance(block, str)
            or block.get("type") == "text"
            or (keep_images and _is_inline_image(block))
        ):
            content.append(block)
        else:
            kind = "image" if block.get("type") in _IMAGE_TYPES else block.get("type", "content")
            content.append({"type": "text", "text": f"[{kind} omitted]"})
    return message.model_copy(update={"content": content})


def latest_images(messages: Sequence[BaseMessage], limit: int) -> list[dict[str, Any]]:
    """The newest inline image blocks in `messages`, oldest first, at most `limit`."""
    found: list[dict[str, Any]] = []
    for message in reversed(messages):
        if isinstance(message.content, str):
            continue
        for block in reversed(message.content):
            if isinstance(block, dict) and _is_inline_image(block):
                found.append(block)
                if len(found) == limit:
                    return found[::-1]
    return found[::-1]


__all__ = ["latest_images", "without_media"]
