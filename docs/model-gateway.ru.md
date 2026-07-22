# Model Gateway

`ModelGateway` — граница интеграции моделей в Muscles-приложении. Business
actions не должны зависеть от OpenAI SDK, локальных inference-библиотек или
формата конкретной модели.

## Архитектура

```text
business action
  -> ModelGateway.invoke(типизированный request)
  -> routing по model alias и capability
  -> provider adapter
  -> внешний SDK/API или локальный runtime внутри процесса
```

Ollama или другой model server не обязательны. Provider может напрямую вызвать
внешний API или загрузить локальную модель в текущий процесс.

## Типизированные capabilities

Facade один, но requests различаются по операции:

- `TextGenerationRequest` -> `TextGenerationResult`;
- `ImageGenerationRequest` -> `ImageGenerationResult`;
- `EmbeddingRequest` -> `EmbeddingResult`;
- в будущем — reranking, transcription, speech synthesis и другие операции.

Не следует делать публичный `execute(dict)`. Provider объявляет поддерживаемые
capabilities, а registry отклоняет неподдерживаемую операцию до вызова backend.

## Конфигурация

Provider описывает SDK/runtime adapter, а model — именованный alias и его
capabilities:

```yaml
modules:
  ai:
    providers:
      local:
        type: llama_cpp
        options:
          model_path: ./models/text.gguf
      images:
        type: openai
        options:
          api_key_env: OPENAI_API_KEY
    models:
      text.local:
        provider: local
        model: text-model
        capabilities: [text.generate]
      image.remote:
        provider: images
        model: dall-e-3
        capabilities: [image.generate]
    defaults:
      text.generate: text.local
```

В action разрешены только настроенные aliases. Произвольные SDK options, URLs и
credentials не должны приходить из публичного payload.

## Optional providers

Тяжёлые зависимости остаются optional:

```bash
pip install "muscles-ai[openai]"
pip install "muscles-ai[llama-cpp]"
```

`python` adapter не требует зависимостей и предназначен для project-owned
Transformers, Diffusers, MLX и других model objects. SDK provider загружается
лениво, поэтому неиспользуемые backend не влияют на запуск пакета.

Например, OpenAI adapter может использовать image model `dall-e-3` и напрямую
вызвать OpenAI API. Это удалённый provider, но для него всё равно не нужен
Ollama или отдельный model server на стороне приложения.

## Совместимость с RAG

Существующий `LLMProvider` подключён к capability `text.generate` через bridge.
`ai.ask` и текущие RAG flows продолжают работать. Новые business actions могут
получить `ModelGateway` из DI и явно выбрать model alias.

```python
from muscles_ai import ModelGateway, TextGenerationRequest

gateway = context.application.container.resolve(ModelGateway)
result = gateway.invoke(
    TextGenerationRequest(prompt="Сделай резюме документа"),
    model="text.local",
)
```

## Владение артефактами

Image и другие binary results нормализуются в `Artifact`. Adapter возвращает
bytes или provider URI, а приложение решает, вернуть, закешировать или сохранить
результат в object storage. Core не пишет файлы и не владеет хранилищем output.
