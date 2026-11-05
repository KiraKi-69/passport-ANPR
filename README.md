# Passport ANPR: YOLOv8 + Label Studio

Проект для разметки и предразметки полей паспорта. Модель YOLOv8 детектирует
рамки полей (`type`, `number`, `surname`, `mrz_line0` и т.д.), а ML-backend
Label Studio возвращает их как готовые предсказания.

## Структура проекта

```
idd.v1i.coco/                 исходный датасет Roboflow в COCO
scripts/prepare_dataset.py    конвертация COCO -> YOLO
notebooks/train_yolov8.ipynb  обучение и оценка YOLOv8
runs/detect/.../weights/      обученные веса (best.pt, last.pt)
datasets/idd_passport/        подготовленный YOLO-датасет (25 классов)
labelstudio_backend/          ML-backend для Label Studio
scripts/start_label_studio.sh запуск Label Studio
```

## 1. Что уже готово

- Обученные веса:
  `runs/detect/idd_passport_fields_yolov8n/weights/best.pt`.
  Копия для backend лежит в репозитории:
  `labelstudio_backend/models/best.pt` (там же `last.pt` и `best.onnx`).
- ML-backend в `labelstudio_backend/`, поддерживает 25 классов из
  `datasets/idd_passport/data.yaml`.
- Конфиг разметки: `labelstudio_backend/labeling_config.xml` (классы совпадают
  с моделью).
- Окружения:
  - `.venv` — Ultralytics, PyTorch, `label-studio-ml`;
  - `.venv-labelstudio` — сам Label Studio.

## 2. Быстрый запуск

Ниже описан локальный запуск без Docker (два venv). Вариант с Label Studio и
backend в контейнерах — в разделе 6.

### 2.1. Терминал 1: Label Studio

```bash
cd /home/kira/python_projects/passport-ANPR

# первый запуск: без автоподключения backend, чтобы получить токен
ML_BACKEND_URL= ./scripts/start_label_studio.sh
```

Откройте `http://localhost:8080`, создайте аккаунт. Затем скопируйте токен:
**Account & Settings -> Personal Access Token**. В Label Studio 1.23+ это
JWT refresh token; backend сам обменивает его на короткоживущий access token.

Запишите токен в `labelstudio_backend/.env`:

```dotenv
LABEL_STUDIO_URL=http://localhost:8080
LABEL_STUDIO_API_KEY=<ваш personal access token>
YOLO_MODEL_PATH=/home/kira/python_projects/passport-ANPR/runs/detect/idd_passport_fields_yolov8n/weights/best.pt
YOLO_DEVICE=0
YOLO_CONF=0.25
YOLO_IOU=0.45
YOLO_IMGSZ=640
LOG_LEVEL=INFO
```

`YOLO_MODEL_PATH` можно оставить пустым — backend сам найдёт
`labelstudio_backend/models/best.pt` или самый новый
`runs/detect/*/weights/best.pt`.

Для машины без GPU укажите `YOLO_DEVICE=cpu`.

### 2.2. Терминал 2: ML-backend

```bash
cd /home/kira/python_projects/passport-ANPR
./labelstudio_backend/start_local.sh
```

Проверка:

```bash
curl http://localhost:9090/health
# {"status":"UP"}
```

Backend слушает `http://localhost:9090`. После первого `predict` в
`labelstudio_backend/backend.log` должно появиться `Loading YOLO model` и
`Model labels: [...]`.

### 2.3. Перезапустите Label Studio с backend

Остановите Label Studio в терминале 1 (`Ctrl+C`) и запустите:

```bash
./scripts/start_label_studio.sh
```

Скрипт автоматически подключит `http://localhost:9090`. Если автоподключение
не сработало, добавьте модель вручную: **Settings -> Model -> Add Model**,
URL `http://localhost:9090`.

Label Studio 1.23+ по умолчанию блокирует localhost и приватные IP для
ML-backend. Для локального запуска в `labelstudio_backend/.env` уже стоят:

```dotenv
LABEL_STUDIO_ML_BLOCK_LOCAL_IP=false
LABEL_STUDIO_SSRF_PROTECTION_ENABLED=false
```

`SSRF_PROTECTION_ENABLED=false` нужен из-за особенностей Label Studio: при
добавлении ML-backend он автоматически создаёт webhook на
`http://localhost:9090/webhook` и проверяет его отдельно, уже без учёта
`ML_BLOCK_LOCAL_IP`. Скрипт запуска Label Studio читает `.env`, поэтому настройки
применяются автоматически. Без них при добавлении модели будет ошибка
`URL resolves to a reserved network address (block: 127.0.0.0/8)`.

### 2.4. Настройка проекта

1. Создайте проект.
2. **Labeling Interface -> Code** — вставьте содержимое
   `labelstudio_backend/labeling_config.xml`.
3. Импортируйте изображения, например из
   `datasets/idd_passport/images/val`.
4. В настройках модели включите **Start labeling using predictions**.
5. Откройте задачу — должны появиться рамки полей.

## 3. Как добавлять и удалять изображения

### Вариант A: через UI (несколько файлов)

1. Откройте проект.
2. Нажмите **Import**.
3. Перетащите изображения или выберите **Upload files**.
4. Нажмите **Import**.

Каждый файл станет задачей с полем `$image`, которое уже используется в
`labeling_config.xml`. Label Studio скопирует загруженные файлы в свой
data-каталог (`~/.local/share/label-studio/media`).

### Вариант B: папка на диске (много изображений)

Label Studio может читать изображения прямо из локальной папки, без загрузки
копий. Это удобно для больших наборов.

1. Создайте папку, например `new_images/`, и положите туда файлы.
2. Разрешите локальные файлы в `.env` (или экспортируйте перед запуском LS):

```dotenv
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/home/kira/python_projects/passport-ANPR
```

3. Перезапустите Label Studio:

```bash
./scripts/start_label_studio.sh
```

4. В проекте: **Settings -> Cloud Storage -> Add Source Storage -> Local Files**.
   - Absolute local path: `/home/kira/python_projects/passport-ANPR/new_images`;
   - включите **Treat every bucket object as a source file**;
   - сохраните и нажмите **Sync**.

Появятся задачи вида
`/data/local-files/?d=new_images/<file>`. Если добавили новые файлы в папку,
нажмите **Sync** ещё раз.

`LOCAL_FILES_DOCUMENT_ROOT` должен быть родительской папкой для хранилища:
путь storage-а обязан находиться внутри него.

### Вариант C: через API

Если локальные файлы уже раздаются Label Studio, задачи можно добавлять пачкой:

```bash
curl -X POST "http://localhost:8080/api/projects/1/import" \
  -H "Authorization: Token $LABEL_STUDIO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '[{"data":{"image":"/data/local-files/?d=new_images/IMG_001.jpg"}}]'
```

Для внешних изображений вместо локального пути используйте URL:

```json
{"data": {"image": "https://example.com/passport.jpg"}}
```

После импорта откройте задачу: backend сделает предсказание. Если рамок нет,
выделите задачи в Data Manager и нажмите **Retrieve predictions** (модель
должна быть подключена в **Settings -> Model**).

### Как удалить задачи

Через UI:

1. Откройте **Data Manager**.
2. Выделите задачи.
3. **Actions -> Delete tasks**.

Чтобы удалить всё сразу — удалите проект: меню проекта -> **Delete Project**.
Вместе с проектом удаляются задачи, аннотации и предсказания.

Через API одну задачу:

```bash
curl -X DELETE "http://localhost:8080/api/tasks/123/" \
  -H "Authorization: Token $LABEL_STUDIO_API_KEY"
```

Несколько задач пачкой:

```bash
curl -X POST "http://localhost:8080/api/dm/actions?id=delete_tasks&project=1" \
  -H "Authorization: Token $LABEL_STUDIO_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"selectedItems":{"all":false,"included":[1,2,3]}}'
```

### Что происходит с файлами на диске

- **Local Files storage**: файлы остаются в `new_images/`. Удаление задач их не
  трогает. Удалите/переместите файлы вручную и нажмите **Sync** в хранилище.
- **Файлы, загруженные через UI**: лежат в
  `~/.local/share/label-studio/media/upload/<project_id>/`. Удаление задач не
  всегда удаляет их с диска. Чтобы освободить место, после удаления задач
  остановите Label Studio и удалите папку проекта:

```bash
rm -rf ~/.local/share/label-studio/media/upload/<project_id>
```

Удалять файлы из `media/upload`, пока задачи существуют, нельзя — изображения
перестанут открываться.

## 4. Полезные команды

```bash
# Label Studio на другом порту
LS_PORT=8081 ./scripts/start_label_studio.sh

# Создать проект автоматически с готовым конфигом
# (сначала создайте пользователя через UI: на пустой базе Label Studio 1.23.2
# падает при --init с AttributeError: quiet_mode)
LS_PROJECT_NAME=passport_anpr \
LABEL_CONFIG_PATH="$PWD/labelstudio_backend/labeling_config.xml" \
./scripts/start_label_studio.sh

# Backend на другом порту
PORT=9091 ./labelstudio_backend/start_local.sh

# Логи backend
tail -f labelstudio_backend/backend.log
```

## 5. Переменные окружения backend

| Переменная | Назначение | По умолчанию |
| --- | --- | --- |
| `LABEL_STUDIO_URL` | URL Label Studio | `http://localhost:8080` |
| `LABEL_STUDIO_API_KEY` | Access token для скачивания изображений | — |
| `YOLO_MODEL_PATH` | Путь к `.pt` весам | автопоиск |
| `YOLO_DEVICE` | `0` для CUDA, `cpu` для CPU | авто |
| `YOLO_CONF` | Порог уверенности | `0.25` |
| `YOLO_IOU` | IoU для NMS | `0.45` |
| `YOLO_IMGSZ` | Размер входа | `640` |
| `LOG_LEVEL` | Уровень логов | `INFO` |

## 6. Запуск Label Studio и backend в Docker

`labelstudio_backend/docker-compose.yml` поднимает два контейнера в одной
сети: Label Studio и YOLO ML-backend. Внутри сети они обращаются друг к другу
по именам сервисов, поэтому `localhost` между ними не используется. Перед
запуском остановите локальный Label Studio, если он занимает порт 8080.

```text
host -> Label Studio:    http://localhost:8080
host -> ML backend:      http://localhost:9090
Label Studio -> backend: http://yolo-backend:9090
backend -> Label Studio: http://label-studio:8080
```

Нужны запущенный Docker Engine (в WSL2 — Docker Desktop с WSL Integration или
Docker Engine внутри WSL) и Compose v2. GPU не обязателен: по умолчанию
собирается CPU-образ без CUDA.

Проверка перед запуском:

```bash
docker info
```

Для GPU дополнительно:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 6.1. Подготовка

```bash
cd /home/kira/python_projects/passport-ANPR/labelstudio_backend
cp .env.example .env   # если файла .env ещё нет
```

В `.env`:

```dotenv
LABEL_STUDIO_API_KEY=
YOLO_MODEL_PATH=
YOLO_DEVICE=cpu
```

- `LABEL_STUDIO_URL` в `.env` можно оставить `http://localhost:8080`: в
  `docker-compose.yml` для контейнера backend он переопределяется на
  `http://label-studio:8080`. Менять значение в `.env` не обязательно.
- `YOLO_MODEL_PATH` оставьте пустым: backend найдёт самый новый
  `/app/runs/detect/*/weights/best.pt`. Каталог `../runs` монтируется в
  `/app/runs`, `./models` — в `/app/models`.
- CPU — режим по умолчанию: `YOLO_DEVICE=cpu`, собирается
  `requirements-cpu.txt`.
- Для GPU есть override-файл `docker-compose.gpu.yml`: он переключает сборку
  на CUDA (`requirements.txt`) и ставит `YOLO_DEVICE=0`.
- Для чистой установки ничего из `~/.local/share/label-studio` переносить не
  нужно: Label Studio хранит данные в named volume `ls-data`.

### 6.2. Сначала запустите Label Studio

```bash
docker compose up -d label-studio
docker compose logs -f label-studio
```

Откройте `http://localhost:8080`, создайте аккаунт. Затем скопируйте токен:
**Account & Settings -> Personal Access Token** и вставьте его в
`labelstudio_backend/.env`:

```dotenv
LABEL_STUDIO_API_KEY=<personal access token>
```

В Label Studio 1.23+ Personal Access Token — это JWT refresh token. Backend
поддерживает его: SDK сам получает по нему access token через
`/api/token/refresh/`. В старых версиях LS этот раздел называется
**Access Token**. Не путайте его с токеном от локального Label Studio: для
Docker-контейнера нужен токен, созданный в этом же контейнере.

Токен обязателен: по нему backend скачивает изображения из Label Studio.
При чистом volume старый токен от локального Label Studio не подходит.
Если backend уже запущен, после изменения `.env` пересоздайте контейнер:

```bash
# CPU
docker compose up -d --force-recreate yolo-backend

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d --force-recreate yolo-backend
```

### 6.3. Запустите backend

CPU (по умолчанию):

```bash
docker compose up -d --build
docker compose ps
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Первая сборка долгая: скачиваются PyTorch и Ultralytics. CPU-образ заметно
меньше CUDA-сборки.

Проверка:

```bash
curl http://localhost:9090/health
# {"status":"UP"}
```

CPU — `cuda.is_available()` должен быть `False`:

```bash
docker compose exec yolo-backend python -c \
  "import torch; print(torch.cuda.is_available())"
```

GPU — должна определиться карта:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec yolo-backend \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 6.4. Подключите модель в Label Studio

1. Создайте проект.
2. **Labeling Interface -> Code** — вставьте содержимое
   `labelstudio_backend/labeling_config.xml`.
3. **Settings -> Model -> Add Model**.
4. URL: `http://yolo-backend:9090` (не `localhost` и не
   `host.docker.internal`).
5. Сохраните модель и включите **Start labeling using predictions**.

Проверить доступность backend из контейнера Label Studio:

```bash
docker compose exec label-studio python -c \
  "import urllib.request; print(urllib.request.urlopen('http://yolo-backend:9090/health').read())"
```

То же через API (после создания проекта):

```bash
TOKEN=<personal access token>
curl -X POST http://localhost:8080/api/ml \
  -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url":"http://yolo-backend:9090","project":1}'
```

### 6.5. Импорт изображений

Через UI: **Import -> Upload files**.

Для папки на диске каталог `../datasets` уже смонтирован в Label Studio как
`/label-studio/files/datasets`. В проекте: **Settings -> Cloud Storage -> Add
Source Storage -> Local Files**:

- Absolute local path: `/label-studio/files/datasets/idd_passport/images/val`;
- включите **Treat every bucket object as a source file**;
- сохраните и нажмите **Sync**.

После открытия задачи backend сделает предсказание. В логах должно появиться:

```bash
docker compose logs -f yolo-backend
# Loading YOLO model: /app/runs/detect/.../best.pt (device=cpu)
# Model labels: ['type', 'code', 'number', ...]
```

На CPU инференс медленнее, чем на GPU. Для ускорения можно уменьшить
`YOLO_IMGSZ` (например, до `512`), но это может немного снизить точность.

### 6.6. Остановка и данные

```bash
docker compose down            # остановить, данные Label Studio сохранятся
docker compose down -v         # удалить БД, медиа и named volume ls-data
```

- `docker compose down -v` необратимо удаляет аккаунты, проекты и разметку
  Label Studio.
- `datasets/`, `runs/` и `labelstudio_backend/models/` — bind mount'ы, команда
  `down -v` их не удаляет.
- Если меняли веса в `runs/`, перезапустите backend:
  `docker compose restart yolo-backend`.

## 7. Создание датасетов

Датасеты создаются скриптом `scripts/prepare_dataset.py` из COCO-разметки
Roboflow в `idd.v1i.coco/`. Скрипт конвертирует COCO bbox в YOLO-формат,
раскладывает изображения и метки по `images/{train,val,test}` и
`labels/{train,val,test}` и создаёт `data.yaml`.

Исходная структура:

```text
idd.v1i.coco/
  train/_annotations.coco.json
  train/*.jpg
  valid/_annotations.coco.json
  valid/*.jpg
  test/_annotations.coco.json
  test/*.jpg
```

### Пресеты классов

- `all` — все 52 класса и исходные split'ы (`train`/`valid`/`test`).
- `passport_fields` — 25 классов полей паспорта; берутся только изображения
  паспортного разворота с рамкой `photo`. Для этого пресета скрипт
  автоматически включает `--resplit`, потому что в исходных `valid`/`test`
  нет паспортных полей.

При `--resplit` split делается по исходному изображению: имя до `.rf.`
считается одной группой, поэтому Roboflow-аугментации одного паспорта не
попадают одновременно в train и val/test.

### Примеры

```bash
# Поля паспорта: 25 классов, новый split 80/10/10
python scripts/prepare_dataset.py \
  --preset passport_fields --resplit \
  --output datasets/idd_passport --force

# Все 52 класса, исходные split'ы
python scripts/prepare_dataset.py \
  --preset all \
  --output datasets/idd --force

# Свой набор классов (имена должны совпадать с COCO-разметкой)
python scripts/prepare_dataset.py \
  --classes "type,number,surname,mrz_line0,mrz_line1,photo" \
  --resplit \
  --output datasets/custom --force

# Не симлинки, а реальные копии изображений
python scripts/prepare_dataset.py \
  --preset passport_fields --resplit \
  --output datasets/idd_passport --copy-images --force
```

### Параметры

| Параметр | Назначение | По умолчанию |
| --- | --- | --- |
| `--source` | Каталог исходного COCO-датасета | `idd.v1i.coco` |
| `--output` | Каталог для YOLO-датасета | `datasets/idd` |
| `--preset` | `all` или `passport_fields` | `all` |
| `--classes` | Список классов через запятую, переопределяет `--preset` | — |
| `--resplit` | Заново разделить изображения на train/val/test | выкл. |
| `--val-frac` | Доля val при `--resplit` | `0.1` |
| `--test-frac` | Доля test при `--resplit` | `0.1` |
| `--seed` | Seed для split | `42` |
| `--copy-images` | Копировать изображения вместо симлинков | выкл. |
| `--force` | Удалить и пересоздать `--output` | выкл. |

По умолчанию изображения в датасете — симлинки на файлы из `idd.v1i.coco`
(в Windows Explorer они могут отображаться как файлы ~1 КБ). Для переноса
датасета на другую машину или в архив используйте `--copy-images`.

После выполнения в `--output` появятся `images/`, `labels/` и `data.yaml`.
Путь к датасету для обучения: `<output>/data.yaml`.

## 8. Обучение и обновление весов

```bash
source .venv/bin/activate

# датасет создаётся по инструкции из раздела 7
jupyter notebook notebooks/train_yolov8.ipynb
```

После обучения новый `best.pt` появится в `runs/detect/<run>/weights/`.
Backend кэширует модель process-wide и проверяет `mtime` файла: при изменении
весов модель перезагрузится автоматически. При смене пути к весам обновите
`YOLO_MODEL_PATH` и перезапустите backend.

`fit()` в backend не реализован: обучение запускается только вручную.

## 9. Диагностика

- **`FileNotFoundError: YOLO weights were not found`** — не найден `best.pt`.
  Проверьте `YOLO_MODEL_PATH`, `labelstudio_backend/models/best.pt` и
  `runs/detect/*/weights/best.pt`.
- **Предсказаний нет** — проверьте `Model labels` в логах; имена классов в
  `labeling_config.xml` должны совпадать с `datasets/idd_passport/data.yaml`.
- **Backend не может скачать изображение / 401** — неверный или пустой
  `LABEL_STUDIO_API_KEY`, либо `LABEL_STUDIO_URL` недоступен из backend.
  Используйте **Personal Access Token** из того же экземпляра Label Studio, к
  которому подключён backend. Токен от локального LS не подходит для Docker-LS.
  JWT access token короткоживущий; Personal Access Token (refresh) backend
  обновляет автоматически.
- **`Runtime error: URL resolves to a reserved network address (block: 127.0.0.0/8)`**
  — Label Studio блокирует localhost/приватные IP для ML-backend и для
  автоматически создаваемого webhook. Добавьте в `labelstudio_backend/.env`:

  ```dotenv
  LABEL_STUDIO_ML_BLOCK_LOCAL_IP=false
  LABEL_STUDIO_SSRF_PROTECTION_ENABLED=false
  ```

  и перезапустите Label Studio через `./scripts/start_label_studio.sh`. Не
  отключайте эту защиту, если Label Studio доступен из недоверенной сети.
- **В Docker Label Studio не видит backend / `localhost` не работает** — внутри
  compose-сети используйте имена сервисов: Label Studio обращается к
  `http://yolo-backend:9090`, backend к `http://label-studio:8080`. Проверьте,
  что оба контейнера в одном compose-проекте и что движок Docker запущен.
- **Порт 8080/9090 занят** — остановите локальный Label Studio/backend или
  измените публикуемые порты в `docker-compose.yml`.
- **`docker: cannot connect to the Docker daemon`** — запустите Docker Desktop
  и включите **WSL Integration** для Ubuntu; проверьте `docker info`.
- **`could not select device driver "nvidia"`** — запущен GPU-режим без GPU или
  NVIDIA runtime. Используйте CPU-запуск: `docker compose up -d --build`.
- **Медленные предсказания на CPU** — нормально для YOLOv8n. Можно уменьшить
  `YOLO_IMGSZ` (например, до `512`) или запустить GPU-override:
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build`.
- **CUDA OOM** — уменьшите `YOLO_IMGSZ` или используйте CPU-режим.
- **Не хватает прав на скрипты** — `chmod +x scripts/*.sh labelstudio_backend/start_local.sh`.

## 10. Ограничения

Backend детектирует только области полей. Распознавание текста (OCR) — отдельный
этап: Tesseract, PaddleOCR, EasyOCR и т.п.
