"""Funções reutilizáveis para a EDA (Parte 1) do desafio Klike."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from klike.data import RAW_DATA_PATH


def missing_value_report(raw_path: Path | str = RAW_DATA_PATH, df_loaded: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compara nulos por coluna: CSV cru (sem conversão de tipos) vs. `df_loaded` (ex.: `load_raw()`).

    Serve para detectar colunas onde uma conversão de tipo (ex.: `astype(bool)`)
    tenha mascarado nulos silenciosamente.
    """
    df_raw = pd.read_csv(raw_path)
    nulls_raw = df_raw.isna().sum().rename("nulos_csv_cru")

    report = nulls_raw.to_frame()
    if df_loaded is not None:
        nulls_loaded = df_loaded.isna().sum().rename("nulos_df_carregado")
        report = report.join(nulls_loaded, how="left")
        report["diferenca"] = report["nulos_df_carregado"] - report["nulos_csv_cru"]

    return report


def subtitle_missingness_analysis(
    raw_path: Path | str = RAW_DATA_PATH,
    col: str = "has_subtitle",
) -> dict[str, pd.DataFrame]:
    """Investiga se os nulos de `col` (lida crua, sem conversão) parecem aleatórios.

    Compara, entre linhas com `col` nulo vs. não nulo: distribuição de
    `platform`, `category`, `format`, e a mediana de `klike_score`, `ctr`,
    `engagement_rate`.
    """
    df_raw = pd.read_csv(raw_path)
    is_null = df_raw[col].isna()

    group_label = is_null.map({True: "nulo", False: "não nulo"})

    dist_tables = {}
    for cat_col in ["platform", "category", "format"]:
        dist = (
            df_raw.groupby(group_label)[cat_col]
            .value_counts(normalize=True)
            .rename("proporcao")
            .mul(100)
            .round(1)
            .unstack(cat_col)
        )
        dist_tables[cat_col] = dist

    medians = df_raw.groupby(group_label)[["klike_score", "ctr", "engagement_rate"]].median().round(3)
    counts = group_label.value_counts().rename("n")

    return {
        "n_por_grupo": counts.to_frame(),
        "distribuicao_platform": dist_tables["platform"],
        "distribuicao_category": dist_tables["category"],
        "distribuicao_format": dist_tables["format"],
        "medianas": medians,
    }


RANGE_RULES: dict[str, str] = {
    "klike_score fora de [0, 100]": "klike_score < 0 or klike_score > 100",
    "music_voice_ratio fora de [0, 1]": "music_voice_ratio < 0 or music_voice_ratio > 1",
    "ctr fora de [0, 1]": "ctr < 0 or ctr > 1",
    "engagement_rate fora de [0, 1]": "engagement_rate < 0 or engagement_rate > 1",
    "impressions negativo": "impressions < 0",
    "clicks negativo": "clicks < 0",
    "conversions negativo": "conversions < 0",
    "spend negativo": "spend < 0",
    "revenue negativo": "revenue < 0",
    "clicks > impressions": "clicks > impressions",
    "conversions > clicks": "conversions > clicks",
}


def check_valid_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada regra de faixa válida, conta quantas linhas violam."""
    rows = []
    for descricao, query in RANGE_RULES.items():
        n_violacoes = len(df.query(query))
        rows.append({"regra": descricao, "linhas_violadas": n_violacoes, "pct": round(100 * n_violacoes / len(df), 2)})
    return pd.DataFrame(rows)


def check_derived_metrics(df: pd.DataFrame, tol: float = 0.01) -> dict[str, pd.DataFrame]:
    """Recalcula ctr, cpc e roas e compara (erro relativo) com o valor informado.

    Linhas com denominador zero (impressions==0 para ctr, clicks==0 para cpc,
    spend==0 para roas) são excluídas do cálculo de erro e contadas separadamente,
    pois a divisão não é definida nesses casos.
    """
    results = {}

    specs = [
        ("ctr", "clicks", "impressions"),
        ("cpc", "spend", "clicks"),
        ("roas", "revenue", "spend"),
    ]

    zero_denom_counts = {}
    rows = []
    for metrica, numerador, denominador in specs:
        is_zero_denom = df[denominador] == 0
        zero_denom_counts[denominador] = int(is_zero_denom.sum())

        valid = df.loc[~is_zero_denom]
        recalculado = valid[numerador] / valid[denominador]
        informado = valid[metrica]

        erro_absoluto = (recalculado - informado).abs()
        informado_zero = informado == 0
        erro_relativo = pd.Series(np.nan, index=valid.index)
        erro_relativo.loc[~informado_zero] = erro_absoluto.loc[~informado_zero] / informado.loc[~informado_zero].abs()

        n_informado_zero = int(informado_zero.sum())
        n_divergente = int((erro_relativo > tol).sum())

        rows.append(
            {
                "metrica": metrica,
                "n_excluido_denominador_zero": zero_denom_counts[denominador],
                "n_informado_zero_excluido_do_pct": n_informado_zero,
                "n_comparavel": len(valid) - n_informado_zero,
                "n_divergente_acima_tol": n_divergente,
                "pct_divergente": round(100 * n_divergente / max(len(valid) - n_informado_zero, 1), 2),
                "erro_absoluto_medio": round(erro_absoluto.mean(), 6),
                "erro_absoluto_max": round(erro_absoluto.max(), 6),
            }
        )

    results["resumo"] = pd.DataFrame(rows)
    results["n_linhas_denominador_zero"] = pd.Series(zero_denom_counts, name="n_linhas")

    watch_time_violacao = df["avg_watch_time_s"] > df["video_duration_s"]
    results["avg_watch_time_maior_que_duracao"] = pd.DataFrame(
        {"n_linhas": [int(watch_time_violacao.sum())], "pct": [round(100 * watch_time_violacao.mean(), 2)]}
    )

    return results


def outlier_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Para cada coluna numérica: skewness, skewness do log1p, % de outliers (IQR)
    e um flag indicando se a distribuição parece log-normal.

    Critério de log-normal: skewness original > 1 E |skewness(log1p(x))| < 0.5 * |skewness original|.
    Só é avaliado para colunas sem valores negativos (log1p exige x >= -1).
    """
    rows = []
    for col in numeric_cols:
        serie = df[col].dropna()
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        pct_outliers = round(100 * ((serie < lower) | (serie > upper)).mean(), 2)

        skew_original = skew(serie)

        pode_logar = (serie >= 0).all()
        skew_log = skew(np.log1p(serie)) if pode_logar else np.nan

        parece_lognormal = bool(
            pode_logar and skew_original > 1 and abs(skew_log) < 0.5 * abs(skew_original)
        )

        rows.append(
            {
                "coluna": col,
                "skewness": round(skew_original, 3),
                "skewness_log1p": round(skew_log, 3) if pode_logar else np.nan,
                "pct_outliers_iqr": pct_outliers,
                "parece_lognormal": parece_lognormal,
            }
        )

    return pd.DataFrame(rows)


def group_size_table(df: pd.DataFrame, col1: str, col2: str, min_n: int = 20):
    """Crosstab col1 x col2 com destaque (via Styler) das células com n < min_n."""
    tabela = pd.crosstab(df[col1], df[col2])

    def _highlight(valor: int) -> str:
        return "background-color: #ffcccc" if valor < min_n else ""

    return tabela.style.map(_highlight)
