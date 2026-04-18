# ─────────────────────────────────────────────────────────────────────────────
# Makefile — MedScan AI
# ─────────────────────────────────────────────────────────────────────────────
# Convenience shortcuts for Docker operations.
# Requires: Docker Desktop or Docker Engine installed.
#
# Usage:
#   make build    Build the Docker image from scratch
#   make run      Start the app (build first if needed)
#   make up       Same as `run` — alias
#   make stop     Stop and remove the running container
#   make logs     Stream live logs from the container
#   make shell    Open a bash shell inside the running container (for debugging)
#   make clean    Remove the Docker image and volumes
#   make local    Run locally without Docker (requires Python + pip installed)
#   make test     Run the full pytest test suite locally
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_NAME  = medscan-ai
CONTAINER   = medscan_ai_app
PORT        = 8501

.PHONY: build run up stop logs shell clean local test

# Build the Docker image
build:
	@echo "Building MedScan AI Docker image..."
	docker build -t $(IMAGE_NAME) .
	@echo "Build complete. Run 'make run' to start."

# Run with docker-compose (recommended — handles volumes and restart policy)
run:
	@echo "Starting MedScan AI at http://localhost:$(PORT) ..."
	docker-compose up --build

up: run

# Stop the running container
stop:
	@echo "Stopping MedScan AI..."
	docker-compose down

# Stream container logs
logs:
	docker-compose logs -f

# Open an interactive shell in the running container (for debugging)
shell:
	docker exec -it $(CONTAINER) /bin/bash

# Remove image, containers, and volumes
clean:
	@echo "Removing MedScan AI containers, images, and volumes..."
	docker-compose down --rmi all --volumes --remove-orphans
	@echo "Cleaned."

# ─── Local (no Docker) ────────────────────────────────────────────────────
# Run locally without Docker. Requires Python 3.9+ and pip.

local:
	@echo "Starting MedScan AI locally..."
	@python3 setup.py && streamlit run app.py

test:
	@echo "Running test suite..."
	python3 -m pytest tests/ -v
