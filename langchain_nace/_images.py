"""Convert images to the entries Drex reads in a request's `images` array."""

from __future__ import annotations

import base64
import binascii
import os
from collections.abc import Mapping
from typing import Any

import nace_sdk

from langchain_nace.errors import NaceImageError

_REMOTE = ("http://", "https://")


def _remote_url_error(url: str) -> NaceImageError:
    return NaceImageError(
        f"Drex doesn't fetch image URLs ({url[:60]}…). Download the image and pass its "
        "bytes, or upload it with `nace_sdk` and pass the uploaded file."
    )


def _from_url(url: str) -> str:
    if url.startswith("data:"):
        return url
    if url.startswith(_REMOTE):
        raise _remote_url_error(url)
    return url


def _from_base64(data: str, mime_type: str | None) -> str:
    return f"data:{mime_type};base64,{data}" if mime_type else data


_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
)


def as_data_url(image: str) -> str:
    """A data URL for base64 image bytes, typed from their first bytes."""
    if image.startswith("data:"):
        return image
    head = base64.b64decode(image[:24] + "=" * (-len(image[:24]) % 4), validate=False)
    mime = next((m for sig, m in _SIGNATURES if head.startswith(sig)), None)
    if mime is None and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        mime = "image/webp"
    return f"data:{mime or 'application/octet-stream'};base64,{image}"


def _is_base64(value: str) -> bool:
    if len(value) % 4:
        return False
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def to_drex_image(item: Any) -> str | dict[str, Any]:
    """Return one `images` entry for Drex.

    Accepts base64, a data URL, bytes, a path, a LangChain image content block (v0
    `source_type`, v1 `base64`/`url`), an `image_url` block, a `workspace_file`
    reference, or an `UploadedFile`. Remote URLs are refused: Drex never fetches them.

    Raises:
        NaceImageError: The item isn't an image Drex can read.

    """
    if isinstance(item, str):
        if item.startswith(("data:", *_REMOTE)):
            return _from_url(item)
        if _is_base64(item):
            return item
        if os.path.exists(item):
            return _from_sdk(item)
        message = f"{item[:60]!r} is not base64, a data URL, or an existing file path."
        raise NaceImageError(message)
    if isinstance(item, bytes | bytearray | os.PathLike):
        return _from_sdk(item)
    if isinstance(item, nace_sdk.UploadedFile | nace_sdk.WorkspaceFileSource):
        return _from_sdk(item)
    if isinstance(item, Mapping):
        return _from_block(item)
    message = f"Unsupported image type: {type(item).__name__}."
    raise NaceImageError(message)


def _from_sdk(item: Any) -> str | dict[str, Any]:
    try:
        return nace_sdk.image_input(item)
    except nace_sdk.NaceError as error:
        raise NaceImageError(str(error)) from error


def _from_block(block: Mapping[str, Any]) -> str | dict[str, Any]:
    kind = block.get("type")
    if kind == "workspace_file":
        return dict(block)
    if kind == "image_url":
        image_url = block.get("image_url")
        url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
        if isinstance(url, str):
            return _from_url(url)
    if kind == "image":
        mime_type = block.get("mime_type")
        if block.get("source_type") == "base64" and isinstance(block.get("data"), str):
            return _from_base64(block["data"], mime_type)
        if isinstance(block.get("base64"), str):
            return _from_base64(block["base64"], mime_type)
        url = block.get("url") or (block.get("data") if block.get("source_type") == "url" else None)
        if isinstance(url, str):
            return _from_url(url)
    message = f"Unsupported image block: {dict(block)!r:.80}"
    raise NaceImageError(message)


__all__ = ["as_data_url", "to_drex_image"]
