"""Turn state containing LangChain messages into the JSON Drex reads.

Drex rejects media inside `state`. Images found in messages are either moved to the
request's `images` array (replaced in the text by "[Image N]"), replaced by a
placeholder, or refused, depending on the caller. Other media (audio, files) is
replaced by a placeholder or refused.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import BaseMessage, convert_to_openai_messages

from langchain_nace._images import to_drex_image
from langchain_nace.errors import NaceImageModelError, NaceMediaError

ImageMode = Literal["lift", "placeholder", "error"]
MediaMode = Literal["placeholder", "error"]

# Roles LangChain accepts in a `{"role", "content"}` dict used in place of a message object.
_MESSAGE_ROLES = frozenset(
    {"user", "human", "assistant", "ai", "system", "developer", "tool", "function"}
)


def _is_dict_message(value: dict[str, Any]) -> bool:
    """A dict message with content blocks; plain-text content needs no conversion."""
    role = value.get("role")
    return (
        isinstance(role, str) and role in _MESSAGE_ROLES and isinstance(value.get("content"), list)
    )


@dataclass
class _Encoder:
    images: ImageMode
    media: MediaMode
    first_index: int
    lifted: list[str | dict[str, Any]] = field(default_factory=list)

    def value(self, value: Any) -> Any:
        if isinstance(value, BaseMessage):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = convert_to_openai_messages(value)
            return self.message(converted)
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                message = "State object keys must be strings."
                raise TypeError(message)
            if _is_dict_message(value):
                return self.dict_message(value)
            return {key: self.value(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return [self.value(item) for item in value]
        if value is None or isinstance(value, str | int | float | bool):
            return value
        message = f"Unsupported state value: {type(value).__name__}."
        raise TypeError(message)

    def dict_message(self, value: dict[str, Any]) -> dict[str, Any]:
        # Normalize only the content blocks, as for a message object; the other keys
        # are the caller's data and stay as they are.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = convert_to_openai_messages({"role": "user", "content": value["content"]})
        content = self.message(converted)["content"]
        return {
            key: content if key == "content" else self.value(item) for key, item in value.items()
        }

    def message(self, message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if isinstance(content, list):
            message = {**message, "content": [self.part(part) for part in content]}
        return message

    def part(self, part: Any) -> Any:
        if not isinstance(part, dict):
            return part
        kind = part.get("type")
        if kind == "text":
            return part
        if kind == "image_url":
            return self.image(part)
        if self.media == "error":
            message = (
                f"Drex reads text and images only; the state holds a `{kind}` content block. "
                "Remove it, or describe it in text."
            )
            raise NaceMediaError(message)
        return {"type": "text", "text": f"[{kind or 'content'} omitted]"}

    def image(self, part: dict[str, Any]) -> dict[str, Any]:
        if self.images == "placeholder":
            return {"type": "text", "text": "[image omitted]"}
        if self.images == "error":
            message = (
                "The state holds an image, but this model reads text only. "
                "Use an image model such as model='drex-max'."
            )
            raise NaceImageModelError(message)
        self.lifted.append(to_drex_image(part))
        return {"type": "text", "text": f"[Image {self.first_index + len(self.lifted) - 1}]"}


def encode_state(
    state: Any,
    *,
    images: ImageMode,
    media: MediaMode = "error",
    first_index: int = 1,
) -> tuple[Any, list[str | dict[str, Any]]]:
    """Return `(state_json, lifted_images)`.

    Args:
        state: A string, JSON object or array, a message, or messages at any depth.
        images: What to do with images in messages: move them to `images` ("lift"),
            replace them with "[image omitted]", or raise.
        media: What to do with audio, files and other media: replace or raise.
        first_index: Number of the first lifted image, after any explicit images.

    Raises:
        TypeError: The root is a bare number, boolean or `None`, or a value isn't JSON.
        NaceImageModelError: An image was found and `images="error"`.
        NaceMediaError: Other media was found and `media="error"`.

    """
    if state is None or isinstance(state, bool | int | float):
        message = "State must be a string, a JSON object or array, or LangChain messages."
        raise TypeError(message)
    encoder = _Encoder(images=images, media=media, first_index=first_index)
    return encoder.value(state), encoder.lifted


def message_text(message: BaseMessage) -> str:
    """The text of a message, with non-text blocks left out."""
    return message.text if isinstance(message.text, str) else str(message.text)


__all__ = ["encode_state", "message_text"]
