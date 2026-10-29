import base64
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_nace._images import as_data_url, to_drex_image
from langchain_nace._state import encode_state
from langchain_nace.errors import NaceImageError, NaceImageModelError, NaceMediaError

FIXTURES = Path(__file__).parents[1] / "fixtures"
RED = base64.b64encode((FIXTURES / "red.png").read_bytes()).decode()
BLUE = base64.b64encode((FIXTURES / "blue.png").read_bytes()).decode()


def test_messages_become_role_content_json() -> None:
    state, images = encode_state(
        [SystemMessage("s"), HumanMessage("h"), AIMessage("a")], images="lift"
    )
    assert state == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "h"},
        {"role": "assistant", "content": "a"},
    ]
    assert images == []


def test_messages_nested_in_json() -> None:
    state, _ = encode_state({"conversation": [HumanMessage("hi")], "tier": "pro"}, images="lift")
    assert state == {"conversation": [{"role": "user", "content": "hi"}], "tier": "pro"}


def test_message_images_are_lifted_in_order() -> None:
    message = HumanMessage(
        content=[
            {"type": "text", "text": "compare"},
            {"type": "image", "base64": RED, "mime_type": "image/png"},
            {"type": "image", "source_type": "base64", "data": BLUE, "mime_type": "image/png"},
        ]
    )
    state, images = encode_state(message, images="lift", first_index=2)
    texts = [part["text"] for part in state["content"]]
    assert texts == ["compare", "[Image 2]", "[Image 3]"]
    assert images == [f"data:image/png;base64,{RED}", f"data:image/png;base64,{BLUE}"]


@pytest.mark.parametrize("role", ["user", "human", "assistant", "system"])
def test_dict_message_images_are_lifted(role: str) -> None:
    message = {
        "role": role,
        "content": [
            {"type": "text", "text": "compare"},
            {"type": "image", "base64": RED, "mime_type": "image/png"},
            {"type": "image", "source_type": "base64", "data": BLUE, "mime_type": "image/png"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED}"}},
        ],
    }
    state, images = encode_state([message], images="lift")
    assert state == [
        {
            "role": role,
            "content": [
                {"type": "text", "text": "compare"},
                {"type": "text", "text": "[Image 1]"},
                {"type": "text", "text": "[Image 2]"},
                {"type": "text", "text": "[Image 3]"},
            ],
        }
    ]
    assert images == [
        f"data:image/png;base64,{RED}",
        f"data:image/png;base64,{BLUE}",
        f"data:image/png;base64,{RED}",
    ]


def test_dict_message_keeps_its_other_keys() -> None:
    message = {
        "role": "user",
        "content": [{"type": "image", "base64": RED, "mime_type": "image/png"}],
        "ts": 1793270000,
        "meta": {"channel": "email"},
    }
    state, images = encode_state({"thread": [message]}, images="lift")
    assert state == {
        "thread": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "[Image 1]"}],
                "ts": 1793270000,
                "meta": {"channel": "email"},
            }
        ]
    }
    assert len(images) == 1


def test_dict_message_images_on_text_model_raise() -> None:
    image = {"type": "image", "base64": RED, "mime_type": "image/png"}
    with pytest.raises(NaceImageModelError, match="drex-max"):
        encode_state([{"role": "user", "content": [image]}], images="error")


@pytest.mark.parametrize(
    "value",
    [
        {"role": "user", "content": "plain text"},
        {"role": "admin", "content": [{"type": "image", "base64": "x"}]},
        {"content": [{"type": "image", "base64": "x"}]},
    ],
    ids=["text-content", "unknown-role", "no-role"],
)
def test_json_that_is_not_a_multimodal_message_is_unchanged(value: dict) -> None:
    state, images = encode_state(value, images="lift")
    assert state == value
    assert images == []


def test_message_images_on_text_model_raise() -> None:
    message = HumanMessage(content=[{"type": "image", "base64": RED, "mime_type": "image/png"}])
    with pytest.raises(NaceImageModelError, match="drex-max"):
        encode_state(message, images="error")


def test_audio_refused_by_default_and_replaced_on_request() -> None:
    message = HumanMessage(content=[{"type": "audio", "base64": "AAAA", "mime_type": "audio/wav"}])
    with pytest.raises(NaceMediaError):
        encode_state(message, images="lift")
    state, _ = encode_state(message, images="lift", media="placeholder")
    assert state["content"] == [{"type": "text", "text": "[input_audio omitted]"}]


def test_scalar_root_rejected() -> None:
    with pytest.raises(TypeError):
        encode_state(5, images="lift")


@pytest.mark.parametrize(
    "item",
    [
        RED,
        f"data:image/png;base64,{RED}",
        {"type": "image", "base64": RED, "mime_type": "image/png"},
        {"type": "image", "source_type": "base64", "data": RED, "mime_type": "image/png"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{RED}"}},
        {"type": "image_url", "image_url": f"data:image/png;base64,{RED}"},
        (FIXTURES / "red.png").read_bytes(),
        FIXTURES / "red.png",
    ],
)
def test_image_shapes_convert(item: object) -> None:
    converted = to_drex_image(item)
    assert isinstance(converted, str)
    assert RED in converted


def test_workspace_file_passes_through() -> None:
    ref = {"type": "workspace_file", "workspace_id": "w", "file_id": "f"}
    assert to_drex_image(ref) == ref


@pytest.mark.parametrize(
    "item",
    [
        "https://example.com/cat.png",
        {"type": "image", "url": "https://example.com/cat.png"},
        "not an image",
        42,
    ],
)
def test_bad_images_refused(item: object) -> None:
    with pytest.raises(NaceImageError):
        to_drex_image(item)


def test_data_url_is_typed_from_bytes() -> None:
    assert as_data_url(RED).startswith("data:image/png;base64,")
