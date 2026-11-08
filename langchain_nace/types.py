"""Question and answer types for Drex.

Names and fields follow `langchain_typesafe.types`, so code written for
`TypeSafeClassifier` runs on `NaceClassifier` after changing the import.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, NotRequired, TypeAlias

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError
from typing_extensions import TypedDict

from langchain_nace.errors import NaceAPIResponseValidationError

_QuestionContent: TypeAlias = str | dict[str, JsonValue] | list[JsonValue]
_StateValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | BaseMessage
    | Sequence["_StateValue"]
    | dict[str, "_StateValue"]
    | None
)

State: TypeAlias = str | BaseMessage | Sequence[_StateValue] | dict[str, _StateValue]
"""Text, JSON, or LangChain messages to evaluate.

Messages may appear at the root or anywhere inside objects and arrays; they are sent
as role/content JSON. Images inside messages are moved to the request's `images`
(Drex Max), because Drex rejects media inside `state`.
"""

ImageLike: TypeAlias = Any
"""An image: base64, a data URL, bytes, a path, a LangChain image content block, an
`image_url` block, a `{"type": "workspace_file", ...}` reference, or an
`UploadedFile` from `nace_sdk`."""


class NoulCriteria(BaseModel):
    """Optional descriptions of the yes and no outcomes of a `Noul`."""

    model_config = ConfigDict(populate_by_name=True)

    true: JsonValue = None
    """What counts as yes, or `None`."""

    false: JsonValue = None
    """What counts as no, or `None`."""


class Noul(BaseModel):
    """A yes/no question; the answer is the probability of yes."""

    type: Literal["noul"] = "noul"
    instructions: _QuestionContent | None = None
    """The question. Optional when `criteria` describes an outcome."""

    criteria: NoulCriteria | None = None
    """Optional descriptions of the yes and no outcomes."""


class Choice(BaseModel):
    """Pick one label from a fixed set."""

    type: Literal["choice"] = "choice"
    criteria: dict[str, JsonValue] = Field(min_length=1)
    """Labels mapped to descriptions; a description may be `None`."""

    instructions: _QuestionContent | None = None
    """The question to answer."""


class Score(BaseModel):
    """Rate the state on an ordered rubric; level `i` scores `i`."""

    type: Literal["score"] = "score"
    criteria: list[JsonValue] = Field(min_length=1)
    """Level descriptions, lowest first."""

    instructions: _QuestionContent | None = None
    """The question to answer."""


Question: TypeAlias = Noul | Choice | Score | Mapping[str, Any]
"""A question object, or a raw question dictionary with any `type`.

Raw dictionaries are sent as given (with `instructions` defaulted to `""`), so a
question type Drex adds later works without a new release of this package.
"""


class ClassifierRequest(TypedDict):
    """Input for one `NaceClassifier` call."""

    state: State
    """What to evaluate."""

    questions: Mapping[str, Question]
    """Question ids mapped to questions. Questions about one state are answered in
    one request."""

    images: NotRequired[Sequence[ImageLike]]
    """Images every question sees, in order, as "Image 1", "Image 2", and so on.
    Image-capable models only (such as `drex-max`)."""


class NoulAnswer(BaseModel):
    """Probability that the answer to a `Noul` is yes."""

    type: Literal["noul"] = "noul"
    noul: float = Field(strict=True)
    """Probability of yes, from 0 to 1."""


class ChoiceAnswer(BaseModel):
    """The selected label with every label's probability."""

    type: Literal["choice"] = "choice"
    choice: str
    """The selected label."""

    probabilities: dict[str, Annotated[float, Field(strict=True)]]
    """Probability of each label."""

    confidence: float = Field(strict=True)
    """How sure the model is; not the same number as the top probability."""


class ScoreAnswer(BaseModel):
    """Expected score with its rubric and level probabilities."""

    type: Literal["score"] = "score"
    score: float = Field(strict=True)
    """Expected level; may fall between two levels."""

    legend: dict[int, JsonValue]
    """The rubric, keyed by level."""

    probabilities: dict[int, Annotated[float, Field(strict=True)]]
    """Probability of each level."""

    confidence: float = Field(strict=True)
    """How sure the model is."""


class RawAnswer(BaseModel):
    """An answer of a type this package version does not model yet."""

    model_config = ConfigDict(extra="allow")

    type: str


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | RawAnswer, Field(union_mode="left_to_right")
]
"""An answer, typed by the question it answers."""


class Usage(BaseModel):
    """Token usage reported by Drex."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    """Billed input tokens."""

    image_tokens: int | None = None
    """Image tokens, already included in `input_tokens`."""

    output_tokens: int | None = None
    """Output tokens (not billed)."""


class ClassifierResponse(BaseModel):
    """Answers and metadata from one Drex request."""

    model: str
    """The model that answered."""

    answers: dict[str, Answer]
    """Every answer, keyed by question id."""

    usage: Usage = Field(default_factory=Usage)
    request_id: str | None = None
    """Quote this in support requests."""

    evaluation_time_ms: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)
    """The response body as Drex sent it, including fields this version does not model."""

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """Yes/no answers keyed by question id."""
        return {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Choice answers keyed by question id."""
        return {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Score answers keyed by question id."""
        return {k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)}

    def validate_answers(self, asked: Mapping[str, Question]) -> None:
        """Require answers of the asked types, labels and rubric lengths.

        Messages contain only question-supplied ids and types, never response text.
        Unknown question types and extra answer ids remain supported.
        """
        for name, question in asked.items():
            label = json.dumps(name[:64], ensure_ascii=True)
            answer = self.answers.get(name)
            if answer is None:
                message = f"Drex returned no answer for question {label}."
                raise NaceAPIResponseValidationError(message)
            wire = to_wire(question)
            kind = wire.get("type")
            expected = {"noul": NoulAnswer, "choice": ChoiceAnswer, "score": ScoreAnswer}
            if kind not in expected:
                continue
            if not isinstance(answer, expected[kind]):
                message = f"Drex's answer to question {label} has the wrong type (expected {kind})."
                raise NaceAPIResponseValidationError(message)
            criteria = wire.get("criteria")
            if (
                isinstance(answer, ChoiceAnswer)
                and isinstance(criteria, Mapping)
                and answer.choice not in criteria
            ):
                message = f"Drex answered question {label} with a choice that was not offered."
                raise NaceAPIResponseValidationError(message)
            if (
                isinstance(answer, ScoreAnswer)
                and isinstance(criteria, list)
                and len(answer.legend) != len(criteria)
            ):
                message = f"Drex answered question {label} with a legend of the wrong length."
                raise NaceAPIResponseValidationError(message)

    @classmethod
    def from_body(
        cls,
        body: Mapping[str, Any],
        request_id: str | None,
        asked: Mapping[str, Question] | None = None,
    ) -> ClassifierResponse:
        """Build a response, optionally validating it against the questions sent.

        Invalid responses raise a sanitized error without response-supplied text.
        """
        try:
            answers = {name: _parse_answer(raw) for name, raw in body["answers"].items()}
            response = cls(
                model=str(body.get("model", "")),
                answers=answers,
                usage=Usage.model_validate(body.get("usage") or {}),
                request_id=request_id or body.get("request_id"),
                evaluation_time_ms=body.get("evaluation_time_ms"),
                raw=dict(body),
            )
        except (ValidationError, TypeError, ValueError, KeyError, AttributeError):
            message = "Drex returned an invalid response."
            raise NaceAPIResponseValidationError(message) from None
        if asked is not None:
            response.validate_answers(asked)
        return response


def _parse_answer(raw: Mapping[str, Any]) -> Answer:
    kind = raw.get("type")
    if kind == "noul":
        answer = NoulAnswer.model_validate(raw)
        _bounded(answer.noul)
        return answer
    if kind == "choice":
        choice = ChoiceAnswer.model_validate(raw)
        _bounded(choice.confidence)
        for probability in choice.probabilities.values():
            _bounded(probability)
        return choice
    if kind == "score":
        data = dict(raw)
        legend = data.get("legend")
        if isinstance(legend, list):
            data["legend"] = dict(enumerate(legend))
        score = ScoreAnswer.model_validate(data)
        if not score.legend or set(score.legend) != set(range(len(score.legend))):
            message = "Drex returned an invalid score legend."
            raise NaceAPIResponseValidationError(message)
        _bounded(score.score, len(score.legend) - 1)
        _bounded(score.confidence)
        for probability in score.probabilities.values():
            _bounded(probability)
        return score
    return RawAnswer.model_validate(raw)


def _bounded(value: float, maximum: float = 1) -> None:
    if not 0 <= value <= maximum:
        message = "Drex returned an answer with an out-of-range number."
        raise NaceAPIResponseValidationError(message)


def _text(value: JsonValue) -> JsonValue:
    """Drex reads text only: send structured content as compact JSON text."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def to_wire(question: Question) -> dict[str, Any]:
    """Return a question as Drex reads it.

    `instructions` is always a string (`""` when omitted), because Drex rejects
    `null` and OpenRouter requires the key. Structured instructions and criteria are
    sent as JSON text. `None` choice descriptions are kept.
    """
    if isinstance(question, Noul):
        wire: dict[str, Any] = {"type": "noul", "instructions": _text(question.instructions) or ""}
        if question.criteria is not None:
            criteria = {
                key: _text(value)
                for key, value in (
                    ("true", question.criteria.true),
                    ("false", question.criteria.false),
                )
                if value is not None
            }
            if criteria:
                wire["criteria"] = criteria
        return wire
    if isinstance(question, Choice):
        return {
            "type": "choice",
            "instructions": _text(question.instructions) or "",
            "criteria": {label: _text(desc) for label, desc in question.criteria.items()},
        }
    if isinstance(question, Score):
        return {
            "type": "score",
            "instructions": _text(question.instructions) or "",
            "criteria": [_text(level) for level in question.criteria],
        }
    if isinstance(question, Mapping):
        wire = dict(question)
        if wire.get("instructions") is None:
            wire["instructions"] = ""
        return wire
    message = f"Unsupported question type: {type(question).__name__}."
    raise TypeError(message)


__all__ = [
    "Answer",
    "Choice",
    "ChoiceAnswer",
    "ClassifierRequest",
    "ClassifierResponse",
    "ImageLike",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "RawAnswer",
    "Score",
    "ScoreAnswer",
    "State",
    "Usage",
    "to_wire",
]
