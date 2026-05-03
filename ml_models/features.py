"""ml_models/features.py

Загрузка признаков и подготовка train/test датасета.
Источник фичей: Postgres table `ml_features` (создаётся Spark job'ом).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import psycopg2

from ml_models.config import DBConfig, FEATURE_COLUMNS, PARQUET_FEATURES_PATH, TARGET_COLUMN


def _connect():
    return psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD,
    )


def load_features_from_postgres() -> pd.DataFrame:
    """Загрузка признаков из Postgres."""

    cols = ["lat_bin", "lon_bin"] + list(FEATURE_COLUMNS) + [TARGET_COLUMN]
    col_sql = ", ".join(cols)
    query = f"SELECT {col_sql} FROM ml_features"

    conn = _connect()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    print(f"Loaded {len(df)} records from Postgres (ml_features)")
    return df


def load_features_from_parquet(path: str = PARQUET_FEATURES_PATH) -> pd.DataFrame:
    """Загрузка признаков из Parquet."""

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} records from Parquet: {path}")
    return df


def prepare_features(
    df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLUMNS
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Подготовка X/y: заполнение пропусков и безопасные приведения."""

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    df[feature_cols] = df[feature_cols].fillna(0)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    y = df[TARGET_COLUMN].fillna(0)
    X = df[feature_cols]

    return X, y, list(feature_cols)


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Статистика по признакам (для отчёта/консоли)."""

    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[cols].describe(include="all")
