.PHONY: help sync run infra infra-down lint format test clean

# Help command to display available options
help:
	@echo "Available commands:"
	@echo "  make sync          - Sync dependencies using uv"
	@echo "  make run           - Run the FastAPI service locally on host using uvicorn"
	@echo "  make infra         - Start pre-existing local infrastructure containers"
	@echo "  make infra-down    - Stop local infrastructure containers"
	@echo "  make lint          - Check code style using black, isort, and flake8"
	@echo "  make format        - Auto-format code using black and isort"
	@echo "  make test          - Run tests using pytest"
	@echo "  make clean         - Clean up cached files and directories"

# Sync dependencies using uv
sync:
	uv sync

# Run the FastAPI service locally on the host machine
run:
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start existing infrastructure containers on the host
infra:
	docker start graphgpt-redis graphgpt-kafka graphgpt-cassandra milvus-etcd milvus-minio milvus-standalone

# Stop existing infrastructure containers on the host
infra-down:
	docker stop graphgpt-redis graphgpt-kafka milvus-standalone milvus-etcd milvus-minio

# Check code formatting and style
lint:
	uv run black --check app tests
	uv run isort --check-only app tests
	uv run flake8 app tests

# Format code automatically
format:
	uv run black app tests
	uv run isort app tests

# Run pytest unit and integration tests
test:
	uv run pytest tests/

# Clean temporary Python cache directories
clean:
	rm -rf __pycache__ .pytest_cache .venv build dist
	find . -type d -name "__pycache__" -exec rm -r {} +
