.PHONY: test integration_tests lint format typecheck check_imports examples

test:
	uv run pytest tests/unit_tests --cov

integration_tests:
	uv run pytest tests/integration_tests -s

lint:
	uv run ruff check langchain_nace tests examples scripts
	uv run ruff format --check langchain_nace

format:
	uv run ruff format langchain_nace
	uv run ruff check --fix langchain_nace

typecheck:
	uv run mypy

check_imports:
	uv run python scripts/check_imports.py $$(find langchain_nace -name '*.py')

examples:
	uv run python examples/support_agent.py
	uv run python examples/langgraph_router.py
