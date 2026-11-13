# YOLOv8 ML backend для Label Studio

Обёртка над `best.pt` для предразметки паспортов: модель возвращает рамки
`rectanglelabels` с классами `type`, `number`, `surname`, `mrz_line0` и т.д.

Бэкенд **только делает инференс**. Обучение (`fit`) не реализовано — модель
дообучается в `notebooks/train_yolov8.ipynb`, после чего бэкенд нужно перезапустить.

Полная инструкция по запуску Label Studio вместе с бэкендом находится в
[корневом README](../README.md).

## 1. Модель

Путь к весам определяется в таком порядке:

1. переменная `YOLO_MODEL_PATH`;
2. `labelstudio_backend/models/best.pt`;
3. самый новый `runs/detect/*/weights/best.pt`.

То есть после обучения достаточно положить/оставить `best.pt` в `runs/detect/...`
и перезапустить бэкенд. Пока весов нет, сервер запускается, но первый `predict`
вернёт `FileNotFoundError` — сначала обучите модель в ноутбуке.

## 2. Запуск без Docker (рекомендуется на этой машине)

Этот вариант не требует Docker: `.venv` уже содержит `label-studio-ml` и
Ultralytics, а `.venv-labelstudio` — сам Label Studio. Это два разных
окружения: backend запускается из `.venv`, Label Studio — из
`.venv-labelstudio`. Для контейнерного запуска см. раздел 3.

Сначала запустите Label Studio в отдельном терминале:

```bash
./scripts/start_label_studio.sh
```

Затем запустите backend:

```bash
export LABEL_STUDIO_URL=http://localhost:8080
export LABEL_STUDIO_API_KEY=<ваш personal access token>

./labelstudio_backend/start_local.sh
```

Проверка:

```bash
curl http://localhost:9090/health
# {"status":"UP"}
```

`LABEL_STUDIO_API_KEY` нужен, чтобы бэкенд мог скачивать изображения,
загруженные в Label Studio. Токен: **Account & Settings → Personal Access
Token**. В LS 1.23+ это JWT refresh token, backend обменивает его на access
token автоматически.

### Если зависимости потерялись

```bash
source .venv/bin/activate
pip install -r labelstudio_backend/requirements.txt
```

## 3. Запуск в Docker

`docker-compose.yml` поднимает два сервиса в одной сети:

- `label-studio` — `http://localhost:8080` на хосте;
- `yolo-backend` — `http://localhost:9090` на хосте.

Внутри compose-сети они обращаются по именам: backend к
`http://label-studio:8080`, Label Studio к `http://yolo-backend:9090`.

Нужны Docker с Compose v2. GPU не обязателен: по умолчанию собирается
CPU-образ (`requirements-cpu.txt`). Для GPU используется
`docker-compose.gpu.yml`; в WSL2 нужен драйвер NVIDIA со стороны Windows,
отдельный драйвер внутри WSL не ставится.

```bash
cd labelstudio_backend
cp .env.example .env
```

В `.env` оставьте `YOLO_MODEL_PATH` пустым: веса монтируются из `../runs`
в `/app/runs`, и backend найдёт `runs/detect/*/weights/best.pt` автоматически.
Host-путь вида `/home/...` внутри контейнера не существует.
Для CPU используйте `YOLO_DEVICE=cpu` (значение по умолчанию в примере).

`LABEL_STUDIO_URL` в `.env` можно оставить `http://localhost:8080`: в
`docker-compose.yml` для backend-контейнера он переопределяется на
`http://label-studio:8080`. Менять его не обязательно.

Сначала запустите только Label Studio:

```bash
docker compose up -d label-studio
```

Откройте `http://localhost:8080`, создайте аккаунт и скопируйте токен из
**Account & Settings -> Personal Access Token**. Вставьте его в `.env`:

```dotenv
LABEL_STUDIO_API_KEY=<personal access token>
```

В Label Studio 1.23+ Personal Access Token — это JWT refresh token. Backend
поддерживает такой токен и сам меняет его на access token. В старых версиях LS
раздел называется **Access Token**. Токен должен быть от того же экземпляра
Label Studio, к которому обращается backend.

Затем соберите и запустите backend.

CPU (по умолчанию):

```bash
docker compose up -d --build
docker compose ps
curl http://localhost:9090/health
# {"status":"UP"}
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
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

Данные Label Studio хранятся в named volume `ls-data`. Команда
`docker compose down -v` удаляет аккаунты, проекты и разметку без возможности
восстановления. Каталоги `datasets/`, `runs/` и `models/` — bind mount'ы,
`down -v` их не удаляет.

## 4. Подключение к Label Studio

1. Создайте проект.
2. В **Labeling Interface** переключитесь в **Code** и вставьте
   содержимое [`labeling_config.xml`](./labeling_config.xml).
3. **Settings → Model → Add Model**.
4. URL:
   - Label Studio и backend в Docker Compose: `http://yolo-backend:9090`;
   - Label Studio на хосте, backend локально: `http://localhost:9090`.
5. Сохраните модель и включите **Start labeling using predictions**.
6. Загрузите изображения. Откройте задачу — появятся предсказания.

Проверить backend из контейнера Label Studio: 

```bash
docker compose exec label-studio python -c \
  "import urllib.request; print(urllib.request.urlopen('http://yolo-backend:9090/health').read())"
```

Логи backend:

```bash
docker compose logs -f yolo-backend
```

При первом предсказании там должны появиться `Loading YOLO model` и
`Model labels: [...]`.

Label Studio 1.23+ блокирует localhost/приватные адреса для ML-backend и
автоматически создаёт webhook на `<backend>/webhook`. Для Docker эти настройки
уже переданы сервису `label-studio` в `docker-compose.yml`:

```dotenv
LABEL_STUDIO_ML_BLOCK_LOCAL_IP=false
LABEL_STUDIO_SSRF_PROTECTION_ENABLED=false
```

Для локального запуска без Docker те же значения должны быть в окружении
Label Studio. Не отключайте защиту, если Label Studio доступен из недоверенной
сети.

Названия классов в `<Label value="...">` должны совпадать с именами из
`datasets/idd_passport/data.yaml`. Если хотите переименовать класс в UI,
используйте mapping:

```xml
<Label value="Номер паспорта" predicted_values="number"/>
```

## 5. Ограничения

- Модель детектирует только рамки. Текст внутри полей (OCR) нужно делать
  отдельным этапом (Tesseract/PaddleOCR/EasyOCR).
- `fit()` не обучает модель: Label Studio не должен запускать training.
- При смене весов перезапустите контейнер/процесс бэкенда.
- Если предсказаний нет — проверьте, что в логах бэкенда загрузились нужные
  `Model labels`, и что имена классов в Label Studio совпадают.
