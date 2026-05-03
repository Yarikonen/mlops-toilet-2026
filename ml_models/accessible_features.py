"""ml_models/accessible_features.py

Загрузка и подготовка датасета для задачи классификации доступности (Accessible).
1 строка = 1 туалет, target = is_accessible (0/1).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import psycopg2

from ml_models.config import DBConfig


ACCESSIBLE_FEATURE_COLUMNS: List[str] = [
    "distance_to_center",
    "gender_encoded",
    "category_encoded",
    "has_changing_place",
    "suburb_toilet_count",
    "grid_toilet_count",
]

ACCESSIBLE_TARGET_COLUMN: str = "is_accessible"


def _connect():
    return psycopg2.connect(
        host=DBConfig.HOST,
        port=DBConfig.PORT,
        database=DBConfig.NAME,
        user=DBConfig.USER,
        password=DBConfig.PASSWORD,
    )


def load_accessible_features_from_postgres() -> pd.DataFrame:
    cols = ["_id", "lat_bin", "lon_bin"] + list(ACCESSIBLE_FEATURE_COLUMNS) + [ACCESSIBLE_TARGET_COLUMN]
    col_sql = ", ".join(cols)
    query = f"SELECT {col_sql} FROM ml_accessible_features"

    conn = _connect()
    try:
        df = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    print(f"Loaded {len(df)} records from Postgres (ml_accessible_features)")
    return df


def load_accessible_features_from_parquet(path: str = "/opt/data/processed/ml_accessible_features") -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} records from Parquet: {path}")
    return df


def prepare_accessible_features(
    df: pd.DataFrame,
    feature_cols: List[str] = ACCESSIBLE_FEATURE_COLUMNS,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    df[feature_cols] = df[feature_cols].fillna(0)

    if ACCESSIBLE_TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {ACCESSIBLE_TARGET_COLUMN}")

    y = df[ACCESSIBLE_TARGET_COLUMN].fillna(0).astype(int)
    X = df[feature_cols]
    return X, y, list(feature_cols)


def get_accessible_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ACCESSIBLE_FEATURE_COLUMNS if c in df.columns]
    return df[cols + [ACCESSIBLE_TARGET_COLUMN]].describe(include="all")
