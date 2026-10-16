"""Carregamento dos dados brutos do desafio Klike."""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "klike_challenge_dataset.csv"

BOOL_COLS = ["is_retargeting", "has_subtitle", "has_cta", "has_hook", "has_face"]
CATEGORY_COLS = ["platform", "category", "objective", "target_audience_age", "format", "text_density"]


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Lê o CSV bruto do desafio com os tipos de coluna corretos.

    - `date` como datetime
    - colunas booleanas como "boolean" (nullable): `astype(bool)` converteria
      `NaN` silenciosamente para `True` em colunas com nulos (ex.: `has_subtitle`)
    - colunas categóricas como category
    """
    df = pd.read_csv(path, parse_dates=["date"])

    for col in BOOL_COLS:
        df[col] = df[col].astype("boolean")

    for col in CATEGORY_COLS:
        df[col] = df[col].astype("category")

    return df
