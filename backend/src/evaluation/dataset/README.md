# PostgreSQL 18 / 19beta RAG-датасет (150 вопросов)

Источники: `PostgreSQL 18.6 Documentation` (3156 стр.) и `PostgreSQL 19beta4 Documentation` (3053 стр., release notes «AS OF 2026-09-14»).

## Состав

| category | кол-во | что это |
|---|---|---|
| `pg18` | 40 | ответ находится в документации 18 |
| `pg19` | 40 | ответ находится в документации 19 (в основном новые возможности) |
| `cross` | 50 | нужны фрагменты из обеих версий (что изменилось, удалено, появилось) |
| `unanswerable` | 20 | ответа в документации нет (проверка на отказ) |

Сложность: hard 88, medium 46, easy 16. Multi-hop: 64 вопроса (50 `cross_version` + 14 `cross_section` — два раздела одной версии). Всего 219 эталонных фрагментов.

## Файлы

- `dataset.jsonl` — основной файл, одна строка = один вопрос.
- `dataset.csv` — то же в плоском виде для просмотра в Excel.
- `corpus/pg18.txt`, `corpus/pg19.txt` — текст PDF (`pdftotext` без `-layout`, страницы разделены `\f`). К нему привязаны смещения.
- `evaluate_retrieval.py` — подсчёт hit@k / recall@k / all@k / MRR.

## Поля `dataset.jsonl`

- `id`, `category`, `target_versions`, `answerable`
- `question_ru`, `question_en` — один и тот же вопрос на двух языках (разметка общая)
- `answer_ru`, `answer_en` — краткий эталонный ответ
- `question_type`: fact / parameter / syntax / how-to / conceptual / comparison / unanswerable
- `difficulty`: easy / medium / hard
- `hop_type`: single / cross_section / cross_version / none; `is_multi_hop`
- `gold_fragments[]`:
  - `version` (`"18"` / `"19"`), `doc_title`, `source_file`
  - `section_path`, `section` — путь по оглавлению PDF
  - `pdf_page_start`, `pdf_page_end` — физический номер страницы PDF (с 1); `printed_page` — номер, напечатанный на странице
  - `char_start`, `char_end` — смещения (в символах) в `corpus/pg<version>.txt`
  - `quote` — дословная цитата (пробелы схлопнуты)
  - `also_in_other_version` — если тот же текст дословно есть в другой версии, здесь его страница и смещения, иначе `null`
- `unanswerable` (только для этой категории): `reason_kind` (out_of_scope / nonexistent / missing_detail), `reason`, `absent_terms`, `nearest_topic` — раздел-«приманка», который ретривер скорее всего вернёт

## Как считать retrieval

```
python evaluate_retrieval.py dataset.jsonl results.jsonl --k 1 3 5 10 [--strict-version]
```

`results.jsonl`: `{"id": "pgq-001", "retrieved": [{"text": "...", "version": "18"}, ...]}` — чанки в порядке ранжирования.
Попадание засчитывается, если чанк содержит цитату (сравнение только по буквам и цифрам, поэтому пробелы, переносы и разметка не мешают) либо не менее 60 % цитаты с любого края — на случай разреза чанкером. Способ нарезки значения не имеет.

- `hit@k` — найден хотя бы один эталонный фрагмент; `recall@k` — доля найденных; `all@k` — найдены все (главная метрика для multi-hop).
- `--strict-version` требует совпадения версии чанка. Без него чанк другой версии принимается, только если текст там дословно тот же.

Ориентир: BM25 по `question_en`, чанки 1500 символов — hit@5 = 0.81 в целом, для `cross` all@5 = 0.40.

## Что учесть

- У 40 из 115 фрагментов категорий `pg18`/`pg19` тот же текст дословно есть в другой версии — смотрите `also_in_other_version`. Если версия у вас задаётся фильтром, используйте `--strict-version`.
- В `cross` у нескольких вопросов фрагмент версии 19 взят из release notes (приложение E.1), когда в самой документации 19 упоминания уже нет (удалённые параметры, переименования).
- Документация 19 — бета: к релизу формулировки и номера страниц могут измениться.
- Каждая цитата проверена скриптом на дословное наличие в тексте; термины из `absent_terms` проверены на отсутствие в обоих документах. Эталонные ответы писались по этим цитатам, отдельной независимой вычитки не было.
