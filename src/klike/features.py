"""Features derivadas de atributos acionáveis do criativo."""

from __future__ import annotations

import numpy as np
import pandas as pd


def n_boas_praticas(df: pd.DataFrame) -> pd.Series:
    """Soma das boas práticas de criativo com evidência de impacto em
    performance REAL (ctr/engagement_rate/watch_rate/roas) na análise da
    Parte 1 (`docs/eda_insights.md`): `has_hook`, `has_face`,
    `text_density != "high"`, `video_duration_s <= 30`.

    **Não inclui** `has_cta`/`has_subtitle`: esses dois só têm evidência de
    elevar o `klike_score`, não a performance real (ver `docs/eda_insights.md`).

    `has_hook`, `has_face` e `text_density` não têm nulos no dataset limpo;
    se `video_duration_s` for nulo, a prática de duração é omitida da soma
    (não contada como "não cumprida") — a linha recebe a soma das práticas
    conhecidas (0 a 3) em vez de 0 a 4.
    """
    hook = df["has_hook"].astype(float)
    face = df["has_face"].astype(float)
    pouco_texto = (df["text_density"] != "high").astype(float)
    duracao_curta = pd.Series(
        np.where(df["video_duration_s"].isna(), np.nan, (df["video_duration_s"] <= 30).astype(float)),
        index=df.index,
    )

    praticas = pd.DataFrame({"hook": hook, "face": face, "pouco_texto": pouco_texto, "duracao_curta": duracao_curta})
    return praticas.sum(axis=1, skipna=True)
