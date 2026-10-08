# RAG System

Classic RAG.
Реализован через FastAPI + PostgreSQL.

Сравнение различных методов улучшения работы классической RAG системы. 
Валидация на датасете составленного из вопросов по новейшим версиям PostgreSQL. (18 и 19beta)

## Возможности

- **Загрузка документов.** `POST /documents/upload` сохраняет файл, извлекает текст из PDF (с очисткой), режет на чанки по разделам и индексирует. Версия документации (`Document.version`) берётся из имени файла (`postgresql-18-*.pdf` → `18`). Статус документа: `pending` → `processing` → `indexed` / `failed`.
- **Поиск.** `POST /retrieval/search` — векторный поиск по косинусному расстоянию (pgvector), опционально с реранкером. Если в вопросе упомянута версия (`PostgreSQL 18`, `18-й`, `в 18 и в 19`), поиск идёт только по документации этой версии; при нескольких версиях выдача делится между ними поровну и чередуется.
- **Генерация ответа.** `POST /llm/generate` строит промпт из найденных чанков (у каждого указаны версия, страницы и раздел) и просит LLM вернуть структурированный JSON: ответ (`answer`), атомарные утверждения (`claims`) со ссылками на источники и список `sources`.
- **Оценка качества.** `POST /eval/run` прогоняет датасет из 150 вопросов через пайплайн и считает метрики поиска и генерации (LLM-судья).

## Стек

| Слой | Технологии |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| БД | PostgreSQL 16 + pgvector, SQLAlchemy 2 (async, asyncpg), Alembic |
| Ингест | pypdf, langchain-text-splitters |
| Модели | эмбеддинги: `voyageai/voyage-4` (1024), LLM: `deepseek/deepseek-v4.1-flash`, реранкер: `voyageai/rerank-3-lite`, судья: `openai/gpt-6-luna-pro` |
| Провайдеры | OpenAI-совместимый API (RouterAI) |
| Клиент | Vite, vanilla JS |
| Инфраструктура | Docker Compose, uv |

## Архитектура

```
upload ─► parser ─► chunker ─► embedding ─► Postgres

query ─► embedding ─► vector search ─► reranker ─► prompt ─► LLM
                                                                                        
```

**Чанкинг.** Текст делится на разделы по оглавлению PDF (outline), а если его нет — по заголовкам в тексте. Каждый раздел режется `RecursiveCharacterTextSplitter` на чанки до 1500 символов с перекрытием 200; слишком короткие куски (< 100 символов) склеиваются с соседними. В эмбеддинг идёт текст чанка с путём раздела в начале (`Part III > Chapter 19 > 19.4.5. I/O`), в БД хранится чистый текст и метаданные: `page_number`, `page_end`, `section`, `section_path`.

**Поиск по версиям** Версия определяется регуляркой, сверяется со списком версий в бд `WHERE documents.version==19`

```
.
├── compose.yaml
├── .env.example
├── docker/postgres/init.sql                      
├── backend/
│   ├── Dockerfile
│   ├── alembic.ini
│   └── src/
│       ├── main.py                   # приложение FastAPI
│       ├── core/config.py            # настройки (pydantic-settings)
│       ├── db/                       # модели, сессия, миграции, запросы
│       ├── ingestion/                # загрузка, парсинг, чанкинг, пайплайн
│       ├── retrieval/                # эмбеддинги, поиск, определение версии, реранкер
│       ├── generator/                # промпт, LLM-клиенты, эндпоинт генерации
│       ├── evaluation/               # датасет, метрики, LLM-судья, эндпоинт /eval/run
│       └── schemas/                  # Pydantic-схемы
└── frontend/                         # клиент на Vite
```

## Быстрый старт

```bash
cp .env.example .env        # заполнить OPENAI_API_KEY 
docker compose up -d --build
```

По поводу ключа: как от OpenAI, так и от совместимого сервиса. Я пользовался моделями предоставляемыми RouterAI.
Также Reranker представлен через API RouterAI. Ключ там используется тот же из переменной окружения OPENAI_API_KEY.

Интерактивная документация: http://localhost:8000/docs

Клиент:

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173, /api проксируется на localhost:8000
```

## API

| Метод | Путь | Назначение |
|---|---|---|
| `POST` | `/documents/upload` | загрузка и индексация документа |
| `POST` | `/retrieval/search` | поиск чанков (`query`, `limit`, `reranker_limit`) |
| `POST` | `/llm/generate` | ответ на вопрос (`query`, `limit`, `reranker_limit`, `debug`) |
| `GET` | `/chunks` | список чанков (`document_ids`, `limit`) |
| `POST` | `/eval/run` | прогон оценки на датасете |
| `GET` | `/health`, `/ready` | проверка API и подключения к БД |

С реранкером `limit` — число кандидатов из векторного поиска, `reranker_limit` — сколько оставить после реранкера. `debug=true` добавляет в `sources` текст чанка и расстояние.

```bash
curl -X POST "http://localhost:8000/llm/generate?query=Какое значение io_method используется по умолчанию в PostgreSQL 18?&limit=20&reranker_limit=10"
```

```json
{
  "answer": "В PostgreSQL 18 значение io_method по умолчанию — worker.",
  "claims": [
    {
      "text": "Значение io_method по умолчанию — worker.",
      "sources": [1]
    }
  ],
  "sources": [
    {
      "id": 1,
      "chunk_id": "f8743cb5-ae48-4866-9ba8-df71af8479ad",
      "doc_id": "cc02b4c7-e396-475f-88cf-5ef2e8e4d1f3",
      "chunk_index": 1529,
      "page_number": 690,
      "page_end": 690,
      "section": "19.4.5. I/O",
      "section_path": [
        "Part III. Server Administration",
        "Chapter 19. Server Configuration",
        "19.4. Resource Consumption",
        "19.4.5. I/O"
      ],
      "used": true,
      "version": "18"
    }
  ]
}
```

## Конфигурация

Настройки читаются из `.env` в корне репозитория ([core/config.py](backend/src/core/config.py)); в Docker Compose — из секции `x-backend-env` в `compose.yaml`.


## Оценка

### Датасет

Сгенерирован LLM. + небольшая ручная проверка

[evaluation/dataset/dataset.jsonl](backend/src/evaluation/dataset/dataset.jsonl) — 150 вопросов по документации PostgreSQL 18 и 19, на русском и английском, с эталонным ответом и цитатами из документации (`gold_fragments`).

| Категория | Вопросов | Что проверяет |
|---|---:|---|
| `pg18` | 40 | вопросы по документации 18 |
| `pg19` | 40 | вопросы по документации 19 |
| `cross` | 50 | сравнение версий: нужно найти фрагменты из обеих |
| `unanswerable` | 20 | ответа в документации нет — ожидается отказ |

### Запуск

```bash
# полный прогон
curl -X POST "http://localhost:8000/eval/run?limit=20&reranker_limit=10"

# только метрики поиска, без LLM и судьи
curl -X POST "http://localhost:8000/eval/run?retrieval_only=true&limit=20&reranker_limit=10"

# подмножество
curl -X POST "http://localhost:8000/eval/run?categories=cross&lang=en&sample_limit=10"
```

### Метрики

| Метрика | Что считает |
|---|---|
| `hit@k` | хотя бы один эталонный фрагмент в top-k |
| `recall@k` | доля эталонных фрагментов в top-k |
| `all_found@k` | все эталонные фрагменты в top-k |
| `mrr` | 1 / позиция первого найденного фрагмента |
| `section_recall`, `page_recall` | найден ли нужный раздел / страница |
| `accuracy`, `partial_rate` | вердикт судьи по сравнению с эталоном: correct / partial / incorrect |
| `refusal_rate` | доля отказов («нет информации») |
| `faithfulness` | доля утверждений ответа (`claims`), подтверждённых найденным контекстом; проверяется судьёй без эталона, отказы не учитываются |

## Результаты

Все прогоны: вопросы на русском (`lang=ru`). Прочерк в столбцах `@10` — в LLM передавалось 5 чанков.
Важно: accuracy между таблицами не сравнимы, так как менял метод оценки в промпте. внутри таблиц можно сравнивать.

### Версии документации

Судья v1, векторный поиск top-5, без реранкера.

| Шаг | hit@1 | hit@5 | all_found@5 | MRR | Accuracy | Partial | Refusal |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 56.9% | 83.8% | 70.8% | 0.671 | 62.0% | 27.3% | 25.3% |
| + версия в промпте | 56.9% | 83.8% | 70.8% | 0.671 | 66.0% | 30.0% | 17.3% |
| + фильтр по версии в поиске | 59.2% | 85.4% | 66.9% | 0.688 | **72.0%** | 24.7% | 18.7% |

### Размер контекста и реранкер

Судья v2 (без штрафа за верные детали сверх эталона, с `faithfulness`), фильтр по версии включён.

| Поиск | Чанков в LLM | hit@1 | hit@5 | hit@10 | all_found@5 | all_found@10 | MRR | Accuracy | Partial | Refusal | Faithfulness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense top-5 | 5 | 59.8% | 85.0% | — | 66.9% | — | 0.694 | 81.6% | 15.6% | 18.4% | 98.9% |
| Dense top-10 | 10 | 58.9% | 86.8% | 94.6% | 52.7% | 81.4% | 0.701 | 89.3% | 8.1% | 16.1% | 99.4% |
| Dense 10 → rerank 5 | 5 | 78.5% | 93.1% | — | 78.5% | — | 0.842 | 83.3% | 12.7% | 17.3% | 99.5% |
| Dense 20 → rerank 10 ¹ | 10 | **80.0%** | **95.4%** | **96.9%** | **83.1%** | **90.8%** | **0.858** | **92.0%** | **6.0%** | **14.0%** | **99.8%** |


### Лучшая конфигурация по категориям

Dense 20 → rerank 10, судья v2.

| Категория | Вопросов | hit@1 | all_found@10 | MRR | Accuracy | Refusal | Faithfulness |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pg18` | 40 | 87.5% | 97.5% | 0.919 | 97.5% | 0.0% | 100.0% |
| `pg19` | 40 | 97.5% | 100.0% | 0.988 | 92.5% | 0.0% | 100.0% |
| `cross` | 50 | 60.0% | 78.0% | 0.706 | 84.0% | 4.0% | 99.4% |
| `unanswerable` | 20 | — | — | — | 100.0% | 95.0% | 100.0% |

### Вопросы-сравнения (`cross`)

| Шаг | Судья | all_found@5 | all_found@10 | Accuracy | Refusal |
|---|---|---:|---:|---:|---:|
| Baseline | v1 | 52.0% | — | 32.0% | 34.0% |
| + версия в промпте | v1 | 52.0% | — | 40.0% | 12.0% |
| + фильтр по версии в поиске | v1 | 38.0% | — | 50.0% | 14.0% |
| Dense top-5 | v2 | 40.0% | — | 64.0% | 14.0% |
| Dense top-10 | v2 | 2.0% ² | 64.0% | 80.0% | 8.0% |
| Dense 10 → rerank 5 | v2 | 58.0% | — | 68.0% | 12.0% |
| Dense 20 → rerank 10 ¹ | v2 | **64.0%** | **78.0%** | **84.0%** | **4.0%** |

¹ с чередованием версий: 18, 19, 18, 19, …

² без чередования первые 5 чанков — из одной версии, поэтому обе почти никогда не попадают в top-5.


## Локальная разработка

```bash
docker compose up -d postgres

cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn src.main:app --reload
```

`.env` в корне репозитория должен указывать на `localhost:5432` (как в `.env.example`). Для запуска оценки вне контейнера нужен `EVAL_DATASET_PATH=src/evaluation/dataset/dataset.jsonl`.

## Миграции

```bash
uv run alembic revision --autogenerate -m "описание"
uv run alembic upgrade head
uv run alembic downgrade -1
```
