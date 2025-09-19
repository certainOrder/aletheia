# OpenAI PGVector API

This project is an API that interfaces between an OpenAI CustomGPT and a local PostgreSQL database using pgvector embeddings. It is designed to facilitate the interaction between the AI model and the database, allowing for efficient storage and retrieval of embeddings.

For local development and smoke test instructions, see docs/DEV_ENVIRONMENT.md.

## Project Structure

```
openai_pgvector_api
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
   cd openai_pgvector_api
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

4. **Set up the PostgreSQL database:**
   - Ensure PostgreSQL is installed and running.
   - Create a database for the project.
   - Update the database connection settings in `app/db/models.py`.

5. **Run the application:**
   ```
   python app/main.py
   ```

## Usage

- The API will be accessible at `http://localhost:8000`.
- You can interact with the API using tools like Postman or curl.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.