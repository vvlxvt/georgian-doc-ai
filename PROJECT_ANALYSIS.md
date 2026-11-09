# PROJECT_ANALYSIS.md

Контекстный файл для возобновления работы над проектом **Georgian Document RAG prototype**.
Актуален на 2026-11-08 (ветка `main`, после изменений контекста и интерфейса чата).

---

## 1. Обзор проекта

Прототип системы вопрос-ответ (RAG) над грузинскими PDF-документами.

| Компонент | Выбор |
|---|---|
| Язык исходных документов | Грузинский (`kat`) |
| Язык ответов и пользовательского интерфейса | Русский (`ru`) |
| Язык кода, идентификаторов, комментариев, тестов, документации | Английский |
| Извлечение текста | PyMuPDF (родной текст), Tesseract OCR fallback (`kat+eng`, `--psm 3`) |
| Хранилище чанков | SQLite + SQLAlchemy 2.0 ORM (`Mapped[]`, `DeclarativeBase`) |
| Векторное хранилище | Qdrant (`qdrant-client`), коллекция `documents`, косинусное расстояние |
| Эмбеддинги | `sentence-transformers`, `BAAI/bge-m3`, 1024 dims, L2-нормализация |
| LLM | Ollama (`/api/generate`, non-streaming) или OpenRouter через raw `httpx` |
| CLI | `main.py` (`preflight`, `index`) |
| Веб-UI | FastAPI + Jinja2 (серверный рендеринг), аддитивный к CLI |

**Языковой split (строго соблюдать):**
- Документы: `kat`
- Пользовательский текст и промпты: `ru`
- Код, тесты, документация: `en`

---

## 2. Стек и ограничения (из AGENTS.md)

- Python 3.12, типовые подсказки PEP 604 (`str \| None`, `StrEnum`).
- ORM: SQLAlchemy 2.0.
- Миграции: Alembic, четыре ревизии (`20261006_01`–`20261108_01`). Схема эволюционирует только через Alembic; `init_db()` — только для тестов и удобства.
- Тесты: pytest 9, маркер `integration` для реальных Ollama/моделей.
- Форматирование: black + ruff, line length 88.
- Запрещено без явного одобрения: LangChain, LlamaIndex, Celery, Redis, OpenRouter структурированный вывод/ретраи/персистентность LLM (см. `LLM_IMPLEMENTATION_QUEUE.md` — не одобрено к реализации).
- LLM-провайдер по умолчанию: `openrouter` (переключается через `LLM_PROVIDER`; `ollama` — альтернатива; без fallback).
- `LLM_TOTAL_TIMEOUT` — общий бюджет запроса, не более 300 сек; `LLM_MAX_ATTEMPTS` — 1–3.
- Tesseract языки: `kat+eng` (по умолчанию).
- UI: локальный-only, bind `127.0.0.1`, без аутентификации/авторизации.

---

## 3. Структура репозитория

```
main.py                 CLI entry point (`preflight`, `index`)
app/
  config.py             Settings (pydantic-settings, frozen, extra="ignore", .env)
  database.py           engine, SessionLocal, init_db()
  models.py             Document, DocumentPage, DocumentChunk, Chat, Message
  repositories.py       DocumentChunkRepository, ChatRepository
  pdf.py                PDFReader / PDFPage (текст + рендер страницы 300 DPI, sha256 PNG)
  ocr.py                OCRProcessor ABC, TesseractOCREngine
  text.py               TextNormalizer, TextChunk
  chunking.py           TextChunker (по символам: абзац -> предложение -> слово, overlap словами)
  embeddings.py         EmbeddingService ABC, HuggingFaceEmbeddingService
  vector_store.py       VectorStore ABC
  qdrant_store.py       QdrantVectorStore (проверка совместимости коллекции, индексы payload)
  indexing.py           ChunkIndexer (embed -> upsert -> удаление старых векторов страницы)
  pipeline.py           DocumentPipeline (hash-гатированный ингест, native vs OCR)
  retrieval.py          Retriever (эмбеддинг запроса -> поиск -> фильтр по score)
  context.py            ContextBuilder (маркеры НАЧАЛО_ИСТОЧНИКА / КОНЕЦ_ИСТОЧНИКА)
  answer.py             AnswerGenerator (русский промпт, валидация цитат [стр. N])
  llm.py                LLMService ABC, OllamaLLMService, OpenRouterLLMService, повторы с тайм-аутом
  rag.py                RAGPipeline (retriever -> контекст -> ответ)
  dependencies.py       lru_cache сервисы, create_document_pipeline(), reset_application_services()
  preflight.py          validate_runtime(): схема БД, Qdrant, LLM-провайдер, языки Tesseract
  web.py                FastAPI factory, статика, проверка origem (защита форм)
  web_dependencies.py   get_session (request-scoped SQLAlchemy сессия)
  web_routes.py         маршруты: /, /documents, /upload, /ask, /chats, /chats/{id}, /health
  chat.py               ChatService (чаты scoped to selected documents; persisting ordered messages + sources_json)
  templates/            Jinja2 (home, documents, upload, ask, chats, chat, base)
  static/css, static/js — стили и маленькие скрипты
alembic/                env.py + versions/ (four revisions, включая переименование схемы чатов)
tests/                  один тест-модуль на app-модуль + test_main.py, conftest.py
REVIEW_QUEUE.md         все 9 задач [x] Completed
LLM_IMPLEMENTATION_QUEUE.md предложенные улучшения (не одобрены к реализации)
.env                    gitignored, реальные секреты
.env.example            отслеживаемый шаблон, должен быть синхронизирован с Settings и tests/test_config.py
```

---

## 4. Потоки данных

### 4.1 Индексация (`main.py index` → `DocumentPipeline`)

1. `sha256` файла → если документ с таким `file_hash` уже существует, используется существующий (дедупликация).
2. Для каждой страницы PyMuPDF извлекает родной текст и рендерит страницу в PNG 300 DPI; `content_hash` = `sha256(png)`.
3. Страница пропускается, если `content_hash` не изменился и статус `PARSED`/`OCR` (идемпотентный повторный ингест).
4. Если длина родного текста < `min_text_length` (по умолчанию 20) → OCR через Tesseract на рендере.
5. Текст нормализуется (`TextNormalizer`), разбивается чанкером (`TextChunker`, максимум `CHUNK_MAX_LENGTH` символов, overlap `CHUNK_OVERLAP` слов).
6. Чанки сохраняются в SQLite (`DocumentChunk`), эмбедятся и апсёртятся в Qdrant (`ChunkIndexer`).
7. **Порядок отказоустойчивости** (SQLite и Qdrant неатомарны): новые векторы индексируются primero, старые векторы страницы удаляются только после успешной индексации; при ошибке откатывается сессия и восстанавливаются предыдущие векторы.

### 4.2 Ответ на вопрос (`RAGPipeline`)

1. Вопрос эмбеддится, ищутся топ-k чанков в Qdrant (опционально по `document_id(s)` и `score_threshold`).
2. Контекст собирается с маркерами `НАЧАЛО_ИСТОЧНИКА [стр. N] [ID источника: S1] ... КОНЕЦ_ИСТОЧНИКА`.
3. Если контекста пуст — фиксированный русский ответ «В предоставленных документах недостаточно информации для ответа.», LLM не вызывается.
4. Иначе LLM отвечает на русском, каждая事実ческая фраза заканчивается `[стр. N]`.

---

## 5. Доменные инварианты (менять только с обоснованием в REVIEW_QUEUE.md)

1. Одна страница на `(document_id, page_number)` — уникальный индекс `uq_document_pages_document_id_page_number`.
2. Длина чанка измеряется **в символах**, а не в токенах.
3. Ингест hash-гатирован: родной текст → `sha256(normalized_text)`; OCR-страницы → `sha256(рендер PNG)`. Неизменённая страница со статусом `PARSED`/`OCR` пропускается.
4. `min_text_length` (по умолчанию 20) решает, нужно ли OCR.
5. SQLite и Qdrant неатомарны — порядок записи/удаления строго определён (см. раздел 4.1).
6. Векторы ключатся по `id` чанка; payload: `document_id`, `page_id`, `page_number`, `chunk_index`, `text`. Смена эмбеддинг-модели требует полной переиндексации.
7. Ответы только по-русски с цитатами `[стр. N]`; модель использует только контекст; текст источника — недоверенные данные (инструкции в документе не выполняются).
8. Префлайт перед индексацией: Qdrant доступен, LLM-провайдер настроен, Tesseract с языками `kat+eng` установлен.

---

## 6. Конфигурация (`.env.example` + `Settings`)

Все настройки через окружение, `Settings` заморожен (`frozen=True`, `extra="ignore"`).

| Переменная | По умолчанию |
|---|---|
| `DATABASE_URL` | `sqlite:///data/app.db` |
| `QDRANT_URL` | `http://localhost:6333` |
| `QDRANT_API_KEY` | пусто |
| `QDRANT_COLLECTION` | `documents` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` |
| `EMBEDDING_SIZE` | `1024` |
| `OLLAMA_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.2:3b` |
| `LLM_PROVIDER` | `openrouter` (или `ollama`) |
| `OPENROUTER_URL` | `https://api.openrouter.ai` |
| `OPENROUTER_API_KEY` | пусто (обязателен при `openrouter`) |
| `OPENROUTER_MODEL` | `gpt-4o-mini` |
| `OPENROUTER_CONNECT_TIMEOUT` | `10.0` |
| `OPENROUTER_READ_TIMEOUT` | `120.0` |
| `OPENROUTER_SUPPORTS_STRUCTURED_OUTPUT` | `false` |
| `OPENROUTER_SUPPORTS_JSON_MODE` | `false` |
| `OPENROUTER_SUPPORTS_STREAMING` | `false` |
| `LLM_MAX_ATTEMPTS` | `2` (1–3) |
| `LLM_TOTAL_TIMEOUT` | `180.0` (max 300) |
| `TESSERACT_LANGUAGES` | `kat+eng` |
| `CHUNK_MAX_LENGTH` | `1000` |
| `CHUNK_OVERLAP` | `100` |
| `MIN_TEXT_LENGTH` | `20` |

Правило: `.env.example` список полей должен совпадать с `Settings` и `tests/test_config.py`. В тестах использовать `Settings(..., _env_file=None)`.

---

## 7. Веб-UI (аддитивный к CLI)

Маршруты в `app/web_routes.py`:
- `GET /` — главная
- `GET /documents` — список документов с пагинацией, статусами страниц, ссылками на оригиналы
- `GET/POST /upload` — загрузка PDF (до 50 MiB, проверка `%PDF-`), индексация через `create_document_pipeline()`, копия оригинала в `data/documents/<hash>.pdf` с верификацией SHA-256
- `GET /documents/{id}/original` — отдача оригинала только при совпадении хеша
- `POST /documents/{id}/delete` — удаление векторов из Qdrant, строк из БД и файла оригинала
- `GET/POST /ask` — вопрос с выбором документов, ответ через `RAGPipeline.answer`
- `GET/POST /chats`, `GET /chats/{id}` — список чатов, создание и просмотр чата с сообщениями
- `POST /chats/{id}/messages` — отправка сообщения через `ChatService` (сохраняет пользовательское + ассистентское сообщения, `sources_json`)
- `POST /chats/delete-selected` — удаление выбранных чатов и их сообщений
- `POST /chats/{id}/delete`, `GET /health`

Защита: проверка `Origin` при мутирующих POST-запросах (same-only). Текст интерфейса на русском. UI bind только `127.0.0.1`.

Шаблоны: `app/templates/` (`base.html`, `home.html`, `documents.html`, `upload.html`, `ask.html`, `chats.html`, `chat.html`).

В правом sidebar у чатов есть чекбоксы и кнопка удаления выбранных чатов; POST обрабатывает `ChatRepository.delete_many()` с ORM-cascade сообщений и same-origin проверкой.

Чат:
- Новые чаты автоматически получают название из имени выбранного документа и даты создания в UTC (`YYYY-MM-DD`); при нескольких документах имена идут в порядке выбора.
- История сообщений отображается в хронологическом порядке; последние пять ходов (не более 6 000 символов) ограничивают контекст последующих вопросов.
- При отправке JavaScript сразу добавляет текст вопроса в `message-list`, прокручивает историю вниз и очищает поле. Для POST сохраняется отдельное скрытое поле; пустые и повторные отправки блокируются.
- После успешного POST поле остаётся пустым; при ошибке валидации введённый текст сохраняется.
- Ответ показывается один раз в общей истории. Дополнительный список страниц под ответом не отображается; цитаты `[стр. N]` остаются в самом тексте.
- История помогает интерпретировать уточняющие вопросы и формировать поисковый запрос, но фактическое обоснование ответа берётся только из выбранных документов.

---

## 8. LLM-слой

- `LLMService` — абстракция; `call(LLMRequest) → LLMResponse`.
- `OllamaLLMService` — `/api/generate`, non-streaming.
- `OpenRouterLLMService` — формат, совместимый с OpenAI Chat Completions, опциональная схема ответа (`response_format`), `provider.require_parameters`.
- `_post_with_error_mapping` — единый шов сети с повторными попытками (экспоненциальная задержка + jitter), общий бюджет `LLM_TOTAL_TIMEOUT`, категории ошибок (`LLMErrorCategory`), маппинг HTTP-кодов. Повтор только при `retryable`, без неконтролируемых вложенных ретраев.
- При `LLM_PROVIDER=ollama` проверяется, что модель установлена в Ollama. При `openrouter` требуется `OPENROUTER_API_KEY`, иначе `preflight` упадёт.
- Никогда не логировать API-ключи, сырые промпты, извлечённый текст документов или сырые ответы модели.

---

## 9. Тестирование

- 243 passed, 4 deselected (`pytest -m "not integration"`; последнее полное выполнение 2026-11-08).
- `ruff check --no-cache .` и `black --check .` проходят. На Windows обычный запуск Ruff ранее падал с внутренней ошибкой package cache при сканировании `alembic/versions`; запуск с `--no-cache` проходит.
- In-memory SQLite (`Base.metadata.create_all` в `conftest.py`), `QdrantClient(":memory:")`.
- Патчинг по пути импорта, например `app.preflight.httpx.get`.
- `@pytest.mark.integration` — реальный Ollama / реальная модель (модель загружается только при `RUN_REAL_MODEL_TESTS=1`, иначе skip, чтобы избежать краша интерпретатора на Windows в `torch`).
- `OllamaLLMService._post` — единственный шов сети; патчить его вместо живого Ollama.
- `tests/test_config.py` очищает окружение перед проверкой дефолтов.
- Тестовые функции на уровне модуля (вложенные `def test_` не собираются).
- Команды:
  - `.\venv\Scripts\python.exe -m pytest -m "not integration"`
  - `.\venv\Scripts\python.exe -m pytest -m integration`
  - `.\venv\Scripts\python.exe -m ruff check .`
  - `.\venv\Scripts\python.exe -m black --check .`

---

## 10. Миграции (Alembic)

- `20261006_01` — создаёт `documents`, `document_pages`, `document_chunks` и уникальный индекс на страницы; идемпотентен (проверяет существующие таблицы/индексы).
- `20261104_01` — историческая ревизия, первоначально добавлявшая `conversations` и `messages`.
- `20261107_01` — историческая ревизия связи многие-ко-многим `conversation_documents`.
- `20261108_01` — приводит активную схему к `chats`, `chat_documents` и `messages.chat_id`.
- `alembic/env.py` читает `settings.database_url` с фоллбеком на `sqlalchemy.url` из `alembic.ini`.
- Команды:
  - `.\venv\Scripts\alembic.exe upgrade head`
  - `.\venv\Scripts\alembic.exe downgrade base`

---

## 11. Очереди и статус

- `REVIEW_QUEUE.md`: все 9 задач **[x] Completed** (миграции, проверка Qdrant, индексация, process-scoped сервисы, устойчивость генерации ответов и т.д.).
- `LLM_IMPLEMENTATION_QUEUE.md`: предложенные улучшения (структурированный вывод, ретраи на уровне провайдера, персистентность запусков LLM, OpenRouter) — **не одобрены к реализации**. Не реализовывать без явного разрешения и разрешения открытых продуктовых решений.

---

## 12. Ключевые файлы и их ответственность

| Файл | Отвечает за |
|---|---|
| `main.py` | CLI: `preflight` и `index` |
| `app/config.py` | Все настройки из окружения |
| `app/database.py` | SQLAlchemy engine/session |
| `app/models.py` | ORM модели: Document, DocumentPage, DocumentChunk, Chat, Message |
| `app/repositories.py` | DocumentChunkRepository, ChatRepository |
| `app/pdf.py` | Чтение PDF, рендер страниц, content-hash |
| `app/ocr.py` | Tesseract OCR |
| `app/text.py` | TextNormalizer |
| `app/chunking.py` | TextChunker (символьное разбиение) |
| `app/embeddings.py` | EmbeddingService, HuggingFaceEmbeddingService |
| `app/vector_store.py` | VectorStore ABC |
| `app/qdrant_store.py` | QdrantVectorStore, совместимость коллекции |
| `app/indexing.py` | ChunkIndexer (эмбендинг -> upsert -> prune) |
| `app/pipeline.py` | DocumentPipeline (конечный ингест PDF) |
| `app/retrieval.py` | Retriever (поиск в Qdrant) |
| `app/context.py` | ContextBuilder (источники с маркерами) |
| `app/answer.py` | AnswerGenerator (русский промпт, валидация цитат) |
| `app/llm.py` | LLMService, Ollama, OpenRouter, повторы |
| `app/rag.py` | RAGPipeline (retriever -> контекст -> ответ) |
| `app/chat.py` | ChatService (чаты scoped к документу) |
| `app/dependencies.py` | lru_cache сервисы, create_document_pipeline, reset_application_services |
| `app/preflight.py` | validate_runtime (БД, Qdrant, LLM, Tesseract) |
| `app/web.py` | FastAPI приложение, статика, защита форм |
| `app/web_dependencies.py` | get_session |
| `app/web_routes.py` | Все веб-маршруты |

---

## 13. Наблюдения / потенциальные риски

- **Секреты**: существует `.env` (gitignored), может содержать реальные ключи OpenRouter.
- **Провайдер по умолчанию `openrouter`** без ключа по умолчанию — `preflight` упадёт с ошибкой, пока не задать `OPENROUTER_API_KEY`.
- **UI локальный-only** — привязка к `127.0.0.1`, нет аутентификации/авторизации.
- **Граница SQLite/Qdrant неатомарна** — при сбое между записью чанков в БД и апсёртом в Qdrant возможны несоответствия; порядок действий в `pipeline.py`/`indexing.py` критичен.
- **Нет кэширования эмбеддингов запроса** — каждый вопрос делает полный forward через `BAAI/bge-m3`.
- `llm.py` большой (~2400 строк) — содержит абстракцию, обоих провайдеров и логику повторов.
- `data/app.db` и `data/app.db-journal` существуют в рабочей директории (локальная БД).
- В `data/input/` есть тестовые PDF: `georgian_scan.pdf`, `georgian_scan_ocr.pdf`, `bad_text.pdf`, `test_page.png`.

---

## 14. Чек-лист при возобновлении работы

1. Проверить `REVIEW_QUEUE.md` и `LLM_IMPLEMENTATION_QUEUE.md` на наличие одобренных задач.
2. Прочитать модули, которые предстоит менять, + их тест-модули.
3. Реализовать в `app/`, затем добавить/обновить соответствующий тест-модуль.
4. Запустить `.\venv\Scripts\python.exe -m ruff check .` и `.\venv\Scripts\python.exe -m black --check .`.
5. Запустить `.\venv\Scripts\python.exe -m pytest -m "not integration"`.
6. При изменении поведения — обновить `README.md`, `README_ru.md` и соответствующую очередь.
7. Не коммитить без запроса пользователя. Не менять историю.
8. Перед индексацией: `.\venv\Scripts\python.exe main.py preflight`.
9. При изменении `Settings` — синхронизировать `.env.example` и `tests/test_config.py`.
10. При изменении схемы БД — новая Alembic-ревизия в стиле(idempotent) существующих.