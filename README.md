# Aletheia: OpenAI-compatible RAG API (FastAPI + pgvector)

This repository provides a FastAPI service exposing OpenAI-compatible endpoints and a RAG flow backed by PostgreSQL with `pgvector`. It includes a lightweight static chat UI and a Docker Compose dev environment with OpenWebUI.

For local development and smoke test instructions, see `docs/DEV_ENVIRONMENT.md`.

For the Phase 1 scope, milestones, and acceptance criteria, see `docs/Implementation_Plan_Phase_1.md`.

For Phase 2 (“real model” testing, reliability, and UX), see `docs/Phase_2_Plan.md`.

## Project Structure

```
aletheia
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── api
│   │   └── routes.py
│   ├── db
│   │   ├── __init__.py
│   │   └── models.py
│   ├── services
│   │   └── openai_service.py
│   └── utils
│       └── embeddings.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd aletheia
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run with Docker Compose (recommended):**
   ```
   cp .env.example .env
   docker compose up -d --build
   ```

   Services:
   - API: `http://localhost:8000`
   - OpenWebUI: `http://localhost:3000`

5. **Run directly (advanced):**
   ```
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

- The API is accessible at `http://localhost:8000`.
- Smoke tests and detailed dev instructions: see `docs/DEV_ENVIRONMENT.md`.
- Phase 1 scope: `docs/Implementation_Plan_Phase_1.md`.
- Phase 2 plan: `docs/Phase_2_Plan.md`.

## Contributing (dev setup)

1. Create a virtual environment and install dev tools:
   ```bash
   make setup
   ```
2. Run linters/type-checkers locally:
   ```bash
   make lint format typecheck
   ```
3. Run tests:
   ```bash
   make test
   ```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.