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
| Модели | emb: voyage-4, llm: Deepseek v4.1 Flash |
| Инфраструктура | Docker Compose, uv |

## Архитектура

```
upload ─► parser ─► chunker ─► embedding ─► Postgres (chunks + vector)

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
│       ├── evaluation/           # валидация
│       ├── ingestion/            # загрузка, парсинг, чанкинг, пайплайн
│       ├── retrieval/            # эмбеддинги и поиск
│       ├── generator/            # промпт, LLM-клиент, эндпоинт генерации
│       └── schemas/              # Pydantic-схемы
└── frontend/                     # пока пусто
```

## Быстрый старт


**1. Запустите проект**

```bash
docker compose up -d --build
```


**2. Документация fastapi**

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
  "answer": "Согласно руководству, чтобы включить станок, необходимо нажать зелёную кнопку «Пуск», расположенную на шкафу управления. Перед этим по пошаговой инструкции следует установить режущий инструмент на шпиндель, отключив напряжение станка. После включения станка нужно включить компьютер и запустить программу NC Studio.",
  "claims": [
    {
      "text": "Для включения станка нужно нажать кнопку «Пуск» зелёного цвета, расположенную на шкафу управления.",
      "sources": [
        1
      ]
    }  
    ...
  ],
  "sources": [
    {
      "id": 1,
      "chunk_id": "74f35463-57aa-4182-90c4-c5d946f6c2a0",
      "doc_id": "37bf11ec-d142-4f88-aab7-3e72d5f24988",
      "chunk_index": 41,
      "page_number": 29,
      "used": true
    }...
  ]
}
```

## Конфигурация

адрес бд в `/backend/.env`

остальные настройки, такие как используемые модели, размер векторного пространства можно найти в `/backend/core/config`


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

##

результаты:

baseline( retrieval limit = 3 )
  "correct": 32,
  "partial": 15,
  "incorrect": 3,
  "accuracy": 0.64,
  "grounded_rate": 0.7,
  "avg_page_recall": 0.9270833333333334,
  "avg_text_recall": 0.90625

baseline( retrieval limit = 5 )
  "correct": 32,
  "partial": 17,
  "incorrect": 1,
  "accuracy": 0.64,
  "grounded_rate": 0.72,
  "avg_page_recall": 0.96875,
  "avg_text_recall": 0.96875,


## TODO

в порядке убывающей важности.
добавить eval, попробовать reranker, сделать поддержку разных документов, добавить фронтенд