"""Images uploaded with nace-sdk and passed by reference (the path for large images)."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import nace_sdk
import pytest

from langchain_nace import Choice

FIXTURES = Path(__file__).parents[1] / "fixtures"
WHICH_RED = {
    "red": Choice(instructions="Which image is red?", criteria={"Image 1": None, "Image 2": None})
}


@pytest.fixture
def uploaded():
    with nace_sdk.NaceClient(
        api_key=os.environ["NACE_API_KEY_LIVE"],
        base_url=os.environ.get("NACE_BASE_URL_LIVE") or None,
    ) as client:
        tag = uuid.uuid4().hex[:8]
        files = {
            color: client.documents.upload(
                FIXTURES / f"{color}.png", path=f"langchain-nace-tests/{tag}-{color}.png"
            )
            for color in ("red", "blue")
        }
        try:
            yield files
        finally:
            for file in files.values():
                client.documents.delete_file(file.workspace_id, file.file_id)


def test_uploaded_files_as_images_in_both_orders(live, request_ids, uploaded) -> None:
    red, blue = uploaded["red"], uploaded["blue"]
    max_ = live("drex-max")
    first = max_.invoke(
        {"state": "Two uploaded images.", "questions": WHICH_RED, "images": [red, blue]}
    )
    second = max_.invoke(
        {"state": "Two uploaded images.", "questions": WHICH_RED, "images": [blue, red]}
    )
    assert first.choices["red"].choice == "Image 1"
    assert second.choices["red"].choice == "Image 2"
    request_ids.append(("uploaded files red,blue", "drex-max", first.request_id))
    request_ids.append(("uploaded files blue,red", "drex-max", second.request_id))


def test_workspace_file_reference_dict(live, uploaded) -> None:
    blue, red = uploaded["blue"], uploaded["red"]
    ref = {"type": "workspace_file", "workspace_id": blue.workspace_id, "file_id": blue.file_id}
    r = live("drex-max").invoke(
        {"state": "Two images.", "questions": WHICH_RED, "images": [ref, red]}
    )
    assert r.choices["red"].choice == "Image 2"
