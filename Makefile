.PHONY: help setup up down logs api-logs worker-logs redis-logs clean test lint format

help:
	@echo "Friend Bot - Makefile Commands"
	@echo "==============================="
	@echo "setup       - Setup the project (create .env)"
	@echo "up          - Start all services with Docker Compose"
	@echo "down        - Stop all services"
	@echo "logs        - Show logs from all services"
	@echo "api-logs    - Show logs from API only"
	@echo "worker-logs - Show logs from worker only"
	@echo "redis-logs  - Show logs from Redis only"
	@echo "clean       - Remove containers and volumes"
	@echo "test        - Run tests (placeholder)"
	@echo "lint        - Run linting (placeholder)"
	@echo "format      - Format code (placeholder)"
	@echo "dev         - Start for development with hot reload"

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env file - update with your credentials"; \
	else \
		echo "✓ .env already exists"; \
	fi

up:
	docker-compose up -d
	@echo "✓ Services started. API: http://localhost:8000"

down:
	docker-compose down

logs:
	docker-compose logs -f

api-logs:
	docker-compose logs -f api

worker-logs:
	docker-compose logs -f worker

redis-logs:
	docker-compose logs -f redis

clean:
	docker-compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned up"

dev:
	docker-compose up

test:
	@echo "Tests not yet implemented"

lint:
	@echo "Linting not yet implemented"

format:
	@echo "Code formatting not yet implemented"
