# RAG System

Учебный RAG-проект: загружаете PDF, система режет его на чанки, строит эмбеддинги, сохраняет в PostgreSQL (pgvector) и отвечает на вопросы по содержимому с привязкой утверждений к источникам.

## Возможности

- **Загрузка документов.** `POST /documents/upload` сохраняет файл на диск, парсит PDF (с очисткой текста), режет на чанки и индексирует. Статус документа: `pending` → `processing` → `indexed` / `failed`.
- **Семантический поиск.** `POST /retrieval/search` возвращает top-k ближайших чанков по косинусному расстоянию (pgvector).
- **Генерация ответа.** `POST /llm/generate` находит релевантные чанки, строит промпт и просит LLM вернуть структурированный JSON: итоговый ответ (`answer`), список утверждений (`claims`), у каждого номера использованных источников, и список `sources` (документ и страница).

> Сейчас парсер поддерживает только PDF.

## Стек

| Слой | Технологии |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| БД | PostgreSQL 16 + pgvector, SQLAlchemy 2 (async, asyncpg), Alembic |
| Ингест | pypdf, langchain-text-splitters (чанки 1000 символов, перекрытие 200) |
| Модели | Ollama: эмбеддинги `embeddinggemma` (768 измерений), LLM `qwen3:4b` |
| Инфраструктура | Docker Compose, uv |

## Архитектура

```
upload ─► parser ─► chunker ─► embedding (Ollama) ─► Postgres (chunks + vector)

query ─► embedding ─► top-k search ─► prompt ─► LLM ─► answer + claims + sources
```

```
.
├── compose.yaml
├── docker/postgres/init.sql      
├── backend/
│   ├── Dockerfile
│   ├── alembic.ini
│   └── src/
│       ├── main.py               # приложение FastAPI
│       ├── core/config.py        # настройки (pydantic-settings)
│       ├── db/                   # модели, сессия, миграции, запросы
│       ├── ingestion/            # загрузка, парсинг, чанкинг, пайплайн
│       ├── retrieval/            # эмбеддинги и поиск
│       ├── generator/            # промпт, LLM-клиент, эндпоинт генерации
│       └── schemas/              # Pydantic-схемы
├── eval/                         # оценка качества (пока заготовка)
└── frontend/                     # пока пусто
```

## Требования

- Docker и Docker Compose
- [Ollama](https://ollama.com) на хосте (для GPU или быстрой работы на CPU; контейнерный вариант описан ниже)
- ~8 ГБ RAM, лучше 16 (для `qwen3:4b` нужно 3-4 ГБ)
- [uv](https://docs.astral.sh/uv/) и Python 3.12, если запускаете backend без Docker

## Быстрый старт (backend в контейнере, Ollama на хосте)

**1. Запустите Ollama и скачайте модели**

```bash
sudo systemctl enable --now ollama   # или просто `ollama serve`
ollama pull embeddinggemma
ollama pull qwen3:4b
```

**2. Разрешите Ollama принимать соединения из контейнеров.** По умолчанию он слушает только `127.0.0.1`:

```bash
sudo systemctl edit ollama
```

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

```bash
sudo systemctl restart ollama
```

> Так Ollama становится доступен из локальной сети без авторизации. В недоверенных сетях ограничьте порт 11434 файрволом.

**3. Запустите проект**

```bash
docker compose up -d --build
```

миграции после бд

**4. Документация fastapi**

Интерактивная документация: <http://localhost:8000/docs>

## Использование

```bash
# 1. Загрузить PDF
curl -F "file=@document.pdf" http://localhost:8000/documents/upload

# 2. Поиск релевантных чанков
curl -X POST "http://localhost:8000/retrieval/search?query=что такое RAG&limit=5"

# 3. Ответ на вопрос
curl -X POST "http://localhost:8000/llm/generate?query=что такое RAG&limit=5"
```

Пример ответа `/llm/generate`:

```json
{
  "answer": "…",
  "claims": [{"text": "…", "sources": [1, 3]}],
  "sources": [
    {"id": 1, "doc_id": "…", "page_number": 4},
    {"id": 3, "doc_id": "…", "page_number": 7}
  ]
}
```

## Конфигурация

адрес бд в `/backend/.env`

остальные настройки, такие как используемые модели, размер векторного пространства можно найти в `/backend/core/config`

## Ollama в контейнере (вместо хостового)

```bash
echo "OLLAMA_HOST=http://ollama:11434" >> .env
docker compose --profile docker-ollama up -d --build
docker compose exec ollama ollama pull embeddinggemma
docker compose exec ollama ollama pull qwen3:4b
```

Для GPU NVIDIA добавьте в сервис `ollama` секцию `deploy.resources.reservations.devices` (нужен `nvidia-container-toolkit`).

## Локальная разработка (backend на хосте)

```bash
docker compose up -d postgres

cd backend
cp .env.example .env        # DATABASE_URL с localhost
uv sync
uv run alembic upgrade head
uv run uvicorn src.main:app --reload
```

Тесты и линтеры:

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```

> `test_ready` требует запущенного Postgres.

## Миграции

```bash
uv run alembic revision --autogenerate -m "описание"
uv run alembic upgrade head
uv run alembic downgrade -1
```

## TODO

в порядке убывающей важности.
добавить eval, попробовать reranker, сделать поддержку разных документов, добавить фронтенд