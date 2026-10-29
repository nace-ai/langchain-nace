import langchain_nace
from langchain_nace.experimental import middleware

EXPECTED = {
    "Choice",
    "ChoiceAnswer",
    "ClassifierRequest",
    "ClassifierResponse",
    "NaceClassifier",
    "NaceRouter",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Score",
    "ScoreAnswer",
    "RawAnswer",
    "nace_conditional_edge",
    "NaceUnprocessableEntityError",
    "NaceImageModelError",
}


def test_public_names() -> None:
    assert set(langchain_nace.__all__) >= EXPECTED
    assert {"NaceAutoModeMiddleware", "NaceModelRouterMiddleware", "ModelChoice"} <= set(
        middleware.__all__
    )


def test_typesafe_names_match() -> None:
    typesafe = {
        "Answer",
        "Choice",
        "ChoiceAnswer",
        "ClassifierRequest",
        "ClassifierResponse",
        "Noul",
        "NoulAnswer",
        "NoulCriteria",
        "Question",
        "Score",
        "ScoreAnswer",
        "State",
        "Usage",
    }
    assert typesafe <= set(langchain_nace.__all__)


def test_typesafe_middleware_names_match() -> None:
    from langchain_nace.experimental import middleware

    assert {"AutoModeMiddleware", "ModelChoice", "ModelRouterMiddleware"} <= set(middleware.__all__)
    assert middleware.AutoModeMiddleware is middleware.NaceAutoModeMiddleware
    assert middleware.ModelRouterMiddleware is middleware.NaceModelRouterMiddleware


def test_every_error_class_is_exported() -> None:
    from langchain_nace import errors

    public = {name for name in errors.__all__ if name.startswith("Nace") and name.endswith("Error")}
    assert public <= set(langchain_nace.__all__)
