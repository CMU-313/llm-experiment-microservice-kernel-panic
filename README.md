# Translator Service

FastAPI microservice that detects whether input text is English and, if needed, translates it to English using an Ollama chat model.

## API

- Endpoint: `GET /`
- Query param: `content` (string)
- Response shape:

```json
{
  "is_english": false,
  "translated_content": "This is a German message"
}
```

Example:

```bash
curl "http://127.0.0.1:5001/?content=Dies%20ist%20eine%20Nachricht%20auf%20Deutsch"
```

## Local Development

### 1) Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run tests

```bash
pytest
```

### 3) Run the API server

```bash
uvicorn app:app --host 0.0.0.0 --port 5001
```

Then open:

- `http://127.0.0.1:5001/?content=Hello`
- `http://127.0.0.1:5001/?content=Hola%20mundo`

## Docker Compose (Translator + Ollama)

The repository includes `docker-compose.yml` with:

- `translator` service on port `5001`
- `ollama` service for model inference

Run:

```bash
docker compose up --build -d
docker compose exec -T ollama ollama pull qwen3:0.6b
```

## Configuration

Environment variables used by `src/translator.py`:

- `OLLAMA_HOST` (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default: `qwen3:0.6b`)
- `OLLAMA_TIMEOUT_CONNECT` (default: `5.0`)
- `OLLAMA_TIMEOUT_READ` (default: `90.0`)

In Docker Compose, `OLLAMA_HOST` is set to `http://ollama:11434`.

## Deployment

GitHub Actions workflow at `.github/workflows/deploy.yml` deploys to your VM over SSH on pushes to `main`, then runs:

```bash
docker compose down
docker compose up --build -d
docker compose exec -T ollama ollama pull qwen3:0.6b
```
