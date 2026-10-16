"""Features derivadas de atributos acionáveis do criativo e pipeline de
pré-processamento do cenário "pré-veiculação" (Parte 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


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


# ---------------------------------------------------------------------------
# Pipeline de features — cenário "pré-veiculação" (Parte 2, modelagem)
# ---------------------------------------------------------------------------

#: Colunas do cenário pré-veiculação (CLAUDE.md, decisão a): só atributos do
#: criativo e contexto, nenhuma métrica de performance.
FEATURE_COLS_PREVEICULACAO = [
    "platform",
    "category",
    "objective",
    "target_audience_age",
    "is_retargeting",
    "video_duration_s",
    "format",
    "has_subtitle",
    "has_cta",
    "has_hook",
    "has_face",
    "text_density",
    "music_voice_ratio",
]

#: Métricas de performance — usadas só no cenário comparativo "com performance".
FEATURE_COLS_PERFORMANCE = [
    "impressions",
    "clicks",
    "ctr",
    "cpc",
    "spend",
    "conversions",
    "revenue",
    "roas",
    "avg_watch_time_s",
    "engagement_rate",
]

_BOOL_COLS_PREVEICULACAO = ["is_retargeting", "has_subtitle", "has_cta", "has_hook", "has_face"]


def montar_X_preveiculacao(df: pd.DataFrame, cenario: str = "preveiculacao") -> pd.DataFrame:
    """Seleciona as colunas do cenário pedido (`"preveiculacao"` ou `"com_performance"`)
    e converte as colunas booleanas (nullable `boolean` OU `object`, caso de um
    DataFrame de uma única linha montado a partir de um dict/Series — ver
    `klike.recommender._campanha_para_X`) para `float64` (nulos -> `np.nan`),
    para compatibilidade com os transformers do sklearn. Usa `.where(notna, nan)`
    antes do `.astype` porque um `pd.NA` "solto" num array `object` (não mais
    dentro de uma `BooleanDtype` real) não é convertido por `.astype("float64")`
    sozinho.
    """
    colunas = FEATURE_COLS_PREVEICULACAO + (FEATURE_COLS_PERFORMANCE if cenario == "com_performance" else [])
    X = df[colunas].copy()
    for col in _BOOL_COLS_PREVEICULACAO:
        coluna = X[col]
        X[col] = coluna.where(coluna.notna(), np.nan).astype("float64")
    return X


class ImputaDuracaoEBoasPraticas(BaseEstimator, TransformerMixin):
    """Transformer customizado: imputa `video_duration_s` pela mediana
    (aprendida SÓ no `fit`, nunca antes do split/CV) e, usando o valor já
    imputado, calcula `n_boas_praticas` = `has_hook` + `has_face` +
    (`text_density` != "high") + (`video_duration_s` imputado <= 30).

    Precisa ficar DEPOIS da imputação de `video_duration_s` dentro do
    Pipeline: sem isso, um vídeo com duração nula contaria como "não segue a
    prática de duração curta" em vez de usar o valor imputado.

    Também adiciona o indicador de nulo de `video_duration_s`
    (`video_duration_s_nulo`) como feature — a imputação de `music_voice_ratio`
    e `has_subtitle` (com seus próprios indicadores) fica a cargo do
    `ColumnTransformer` seguinte no pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.mediana_video_duration_s_ = float(X["video_duration_s"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        indicador_nulo = X["video_duration_s"].isna().astype(float)
        duracao_imputada = X["video_duration_s"].fillna(self.mediana_video_duration_s_)
        X["video_duration_s"] = duracao_imputada
        X["video_duration_s_nulo"] = indicador_nulo
        X["n_boas_praticas"] = (
            X["has_hook"].astype(float)
            + X["has_face"].astype(float)
            + (X["text_density"] != "high").astype(float)
            + (duracao_imputada <= 30).astype(float)
        )
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        return np.asarray(base + ["video_duration_s_nulo", "n_boas_praticas"])


def build_preprocessador_preveiculacao(incluir_performance: bool = False, incluir_n_boas_praticas: bool = True) -> ColumnTransformer:
    """`ColumnTransformer` final do cenário pré-veiculação (aplicado DEPOIS de
    `ImputaDuracaoEBoasPraticas`): one-hot das categóricas, ordinal de
    `text_density`, binárias como estão, imputação + indicador para
    `has_subtitle`/`music_voice_ratio`, `log1p` de `video_duration_s`
    (assimétrica — ver EDA, seção 1.7), e passthrough do indicador de nulo de
    duração e de `n_boas_praticas`.

    `incluir_performance=True` adiciona um ramo de `log1p` para as métricas de
    performance (cenário comparativo "com performance"). `incluir_n_boas_praticas=False`
    remove `n_boas_praticas` das features (usado só na ablação da seção 2 do notebook).
    """
    colunas_passthrough_duracao = ["video_duration_s_nulo", "n_boas_praticas"] if incluir_n_boas_praticas else ["video_duration_s_nulo"]
    transformers = [
        (
            "onehot_categoricas",
            OneHotEncoder(drop="first", handle_unknown="ignore"),
            ["platform", "category", "objective", "target_audience_age", "format"],
        ),
        ("ordinal_texto", OrdinalEncoder(categories=[["low", "medium", "high"]]), ["text_density"]),
        ("binarias", "passthrough", ["is_retargeting", "has_cta", "has_hook", "has_face"]),
        ("subtitle", SimpleImputer(strategy="most_frequent", add_indicator=True), ["has_subtitle"]),
        ("music_voice", SimpleImputer(strategy="median", add_indicator=True), ["music_voice_ratio"]),
        ("log_duracao", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), ["video_duration_s"]),
        ("duracao_nula_e_boas_praticas", "passthrough", colunas_passthrough_duracao),
    ]
    if incluir_performance:
        transformers.append(("log_performance", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), FEATURE_COLS_PERFORMANCE))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline_preveiculacao(modelo, incluir_performance: bool = False, incluir_n_boas_praticas: bool = True) -> Pipeline:
    """Pipeline completo (features + modelo) do cenário pré-veiculação (ou
    "com performance", se `incluir_performance=True`). Nenhum pré-processamento
    é aplicado fora deste Pipeline — evita vazamento de dados (CLAUDE.md, regra c).
    `incluir_n_boas_praticas=False` remove `n_boas_praticas` das features (ablação)."""
    return Pipeline(
        [
            ("duracao_e_boas_praticas", ImputaDuracaoEBoasPraticas()),
            (
                "preprocessamento",
                build_preprocessador_preveiculacao(
                    incluir_performance=incluir_performance, incluir_n_boas_praticas=incluir_n_boas_praticas
                ),
            ),
            ("modelo", modelo),
        ]
    )
