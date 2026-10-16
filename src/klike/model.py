"""Comparação de modelos, avaliação e persistência do modelo de `klike_score` (Parte 2)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_validate

from klike.data import load_clean
from klike.features import build_pipeline_preveiculacao, montar_X_preveiculacao

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "klike_model.joblib"

SCORING = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


def _linha_resumo(nome: str, rmse: np.ndarray, mae: np.ndarray, r2: np.ndarray) -> dict:
    return {
        "modelo": nome,
        "rmse_media": -rmse.mean(),
        "rmse_dp": rmse.std(),
        "mae_media": -mae.mean(),
        "mae_dp": mae.std(),
        "r2_media": r2.mean(),
        "r2_dp": r2.std(),
    }


def avaliar_cv_simples(nome: str, modelo, X: pd.DataFrame, y: pd.Series, cv, **kwargs_pipeline) -> dict:
    """Roda `cross_validate` (sem busca de hiperparâmetros — usado para o
    `DummyRegressor`, o `RidgeCV` e para a ablação de `n_boas_praticas`) e
    retorna uma linha de resumo (RMSE/MAE/R² média ± desvio padrão).
    `**kwargs_pipeline` repassa para `build_pipeline_preveiculacao`
    (ex.: `incluir_n_boas_praticas=False`)."""
    pipeline = build_pipeline_preveiculacao(modelo, **kwargs_pipeline)
    resultado = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=[SCORING["rmse"], SCORING["mae"], SCORING["r2"]],
    )
    return _linha_resumo(
        nome,
        resultado[f"test_{SCORING['rmse']}"],
        resultado[f"test_{SCORING['mae']}"],
        resultado[f"test_{SCORING['r2']}"],
    )


def avaliar_cv_grid(nome: str, modelo, grade: dict, X: pd.DataFrame, y: pd.Series, cv) -> tuple[dict, GridSearchCV]:
    """Roda `GridSearchCV` (grade pequena, escolhida por RMSE) na mesma `cv`
    usada para os demais candidatos, e retorna a linha de resumo da MELHOR
    combinação (RMSE/MAE/R² média ± desvio padrão) + o `GridSearchCV` ajustado."""
    pipeline = build_pipeline_preveiculacao(modelo)
    grade_prefixada = {f"modelo__{k}": v for k, v in grade.items()}
    busca = GridSearchCV(
        pipeline, grade_prefixada, cv=cv,
        scoring=[SCORING["rmse"], SCORING["mae"], SCORING["r2"]],
        refit=SCORING["rmse"],
    )
    busca.fit(X, y)
    i = busca.best_index_
    r = busca.cv_results_
    linha = {
        "modelo": nome,
        "rmse_media": -r[f"mean_test_{SCORING['rmse']}"][i],
        "rmse_dp": r[f"std_test_{SCORING['rmse']}"][i],
        "mae_media": -r[f"mean_test_{SCORING['mae']}"][i],
        "mae_dp": r[f"std_test_{SCORING['mae']}"][i],
        "r2_media": r[f"mean_test_{SCORING['r2']}"][i],
        "r2_dp": r[f"std_test_{SCORING['r2']}"][i],
    }
    return linha, busca


def train_and_save(modelo, caminho: Path | str = MODEL_PATH):
    """Treina o Pipeline completo (pré-processamento + `modelo`, já com os
    hiperparâmetros escolhidos) no dataset COMPLETO (`load_clean()`, sem
    split — chamar só depois de concluída a avaliação no conjunto de teste) e
    salva em `caminho` (`models/` está no `.gitignore`; rode esta função para
    recriar o arquivo). A Parte 3 recarrega esse Pipeline com `joblib.load`.
    """
    df = load_clean()
    X = montar_X_preveiculacao(df)
    y = df["klike_score"]

    pipeline = build_pipeline_preveiculacao(modelo)
    pipeline.fit(X, y)

    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, caminho)
    return pipeline
