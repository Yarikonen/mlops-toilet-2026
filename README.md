# MLOps Toilet Pipeline

Пайплайн для обработки данных о общественных туалетах с использованием Apache Airflow, Spark и PostgreSQL.

## Архитектура

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  Data Gov   │────▶│   Airflow    │────▶│    Spark    │────▶│ Postgres │
│  API        │     │   (Orchestr) │     │  (Process)  │     │  (Store) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
```

## Компоненты

| Сервис | Порт | Описание |
|--------|------|----------|
| Airflow Web UI | 8080 | Веб-интерфейс для управления DAG |
| Postgres | 5432 | База данных Airflow + хранилище данных |
| Spark Master | 8081, 7077 | Spark кластер |

## Быстрый старт

### 1. Запуск инфраструктуры

```bash
docker-compose up -d --build
```

### 2. Инициализация Airflow

```bash
# Дождаться запуска всех сервисов
docker-compose logs -f airflow-webserver

# Создать пользователя Airflow
docker-compose exec airflow-webserver airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

### 3. Доступ к интерфейсам

- **Airflow**: http://localhost:8080 (admin/admin)
- **Spark Master**: http://localhost:8081

### 4. Подключение к Postgres

```bash
# Через psql
docker-compose exec postgres psql -U airflow -d airflow

# Или через любой PostgreSQL клиент
# Host: localhost:5432
# Database: airflow
# User: airflow
# Password: airflow
```

## DAGs

### toilet_incremental_pipeline

Основной пайплайн для инкрементальной подгрузки и обработки данных:

1. **ingest_incremental_data** — загрузка новых данных из API
2. **merge_incremental_data** — объединение с существующими данными
3. **create_tables** — создание таблиц в Postgres
4. **run_spark_job** — обработка и анализ данных через Spark
5. **upload_raw_to_postgres** — загрузка сырых данных в Postgres
6. **upload_processed_to_postgres** — загрузка обработанных данных в Postgres

**Расписание**: ежедневно (`@daily`)

## Структура проекта

```
.
├── docker-compose.yml          # Конфигурация Docker сервисов
├── Dockerfile.airflow          # Кастомный образ Airflow с зависимостями
├── airflow/
│   ├── dags/
│   │   ├── toilet_pipeline.py         # Оригинальный DAG
│   │   └── toilet_incremental_pipeline.py  # Инкрементальный DAG
│   ├── logs/                   # Логи Airflow
│   ├── plugins/                # Плагины Airflow
│   └── requirements.txt        # Python зависимости для Airflow
├── spark_jobs/
│   └── job.py                  # Spark job для обработки
├── data/
│   ├── toilets.csv             # Исходные данные
│   └── processed/              # Обработанные данные (Parquet)
├── .gitignore
└── README.md
```

## Схема базы данных (Postgres)

### Таблицы

#### `toilets_raw` — сырые данные
| Колонка | Тип | Описание |
|---------|-----|----------|
| _id | TEXT | Уникальный ID туалета (PK) |
| latitude | DOUBLE PRECISION | Широта |
| longitude | DOUBLE PRECISION | Долгота |
| name | TEXT | Название |
| address | TEXT | Адрес |
| suburb | TEXT | Район |
| postcode | TEXT | Почтовый индекс |
| group_name | TEXT | Группа |
| category | TEXT | Категория |
| accessible | TEXT | Доступность для инвалидов |
| changing_place | TEXT | Пеленальный столик |
| gender | TEXT | Пол |
| created_at | TIMESTAMP | Дата создания записи |
| updated_at | TIMESTAMP | Дата обновления записи |

#### `toilets_grid_stats` — статистика по сетке
| Колонка | Тип | Описание |
|---------|-----|----------|
| id | SERIAL | ID записи (PK) |
| lat_bin | BIGINT | Bin широты |
| lon_bin | BIGINT | Bin долготы |
| toilet_count | BIGINT | Количество туалетов |
| z_score | DOUBLE PRECISION | Z-score для аномалий |

#### `toilets_anomalies` — аномалии (низкая плотность)
Аналогично `toilets_grid_stats`, содержит grid-ячейки с z_score < -2

#### `toilets_hotspots` — горячие точки (высокая плотность)
Аналогично `toilets_grid_stats`, содержит grid-ячейки с z_score > 2

#### `toilets_suburb_stats` — статистика по районам
| Колонка | Тип | Описание |
|---------|-----|----------|
| id | SERIAL | ID записи (PK) |
| suburb | TEXT | Название района |
| postcode | TEXT | Почтовый индекс |
| avg_lat | DOUBLE PRECISION | Средняя широта |
| avg_lon | DOUBLE PRECISION | Средняя долгота |
| accessibility_rate | DOUBLE PRECISION | Доля доступных туалетов |
| toilet_count | BIGINT | Количество туалетов |

## Обработка данных

### Предобработка (Spark)
- Очистка от пустых значений
- Удаление дубликатов
- Валидация координат

### Feature Engineering
- Пространственное бинирование (grid 0.01°)
- Расчет доступности для людей с ограниченными возможностями
- Классификация по типу пола

### Аналитика
- Агрегация по grid с подсчетом туалетов
- Расчет Z-score для обнаружения аномалий
- Статистика по районам (Suburb)

## Примеры SQL запросов

### Найти горячие точки (высокая плотность туалетов)
```sql
SELECT lat_bin, lon_bin, toilet_count, z_score
FROM toilets_hotspots
ORDER BY toilet_count DESC
LIMIT 10;
```

### Статистика по районам
```sql
SELECT suburb, toilet_count, accessibility_rate
FROM toilets_suburb_stats
ORDER BY toilet_count DESC
LIMIT 20;
```

### Найти районы с низкой доступностью
```sql
SELECT suburb, toilet_count, accessibility_rate
FROM toilets_suburb_stats
WHERE accessibility_rate < 0.5
ORDER BY accessibility_rate ASC;
```

### Карта туалетов (для визуализации)
```sql
SELECT latitude, longitude, name, suburb, accessible
FROM toilets_raw
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
```

## Мониторинг

### Проверка статуса сервисов

```bash
docker-compose ps
```

### Логи

```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f airflow-scheduler
docker-compose logs -f spark-master
```

### Проверка данных в Postgres

```bash
docker-compose exec postgres psql -U airflow -d airflow -c "SELECT count(*) FROM toilets_raw;"
docker-compose exec postgres psql -U airflow -d airflow -c "SELECT count(*) FROM toilets_grid_stats;"
```

## Остановка

```bash
docker-compose down
# С удалением volumes (осторожно: удалит все данные!)
docker-compose down -v
```

## Требования к ресурсам

- RAM: минимум 4GB (рекомендуется 8GB)
- CPU: 2+ ядра
- Disk: ~1GB для данных + место под логи

## Расширение

### Добавление новых DAGs

1. Создайте файл в `airflow/dags/`
2. Airflow автоматически обнаружит новый DAG

### Добавление зависимостей Python

Добавьте пакеты в `airflow/requirements.txt` и пересоберите:

```bash
docker-compose up -d --build
```

## Troubleshooting

### Spark job не запускается

```bash
# Проверить логи Spark worker
docker-compose logs spark-worker

# Проверить доступность Postgres
docker-compose exec postgres pg_isready -U airflow
```

### Ошибки подключения к Postgres

```bash
# Проверить доступность БД
docker-compose exec postgres psql -U airflow -d airflow -c "SELECT 1;"

# Проверить таблицы
docker-compose exec postgres psql -U airflow -d airflow -c "\dt"
```

### DAG не появляется в Airflow

```bash
# Проверить логи scheduler
docker-compose logs airflow-scheduler

# Проверить синтаксис DAG
docker-compose exec airflow-webserver python -m py_compile /opt/airflow/dags/toilet_incremental_pipeline.py
```

## License

MIT
