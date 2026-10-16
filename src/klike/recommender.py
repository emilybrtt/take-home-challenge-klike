"""Recommendations Engine (Parte 3): recomendações acionáveis e quantificadas
para uma campanha, combinando o modelo preditivo (Parte 2) e os achados de
performance da EDA (Parte 1).

Duas fontes de quantificação, sempre explícitas (CLAUDE.md, regra e):
- **Impacto no score**: "what-if" com o modelo salvo — sempre disponível.
- **Impacto em performance** (ctr/engagement_rate/watch_rate): lift da mediana
  com IC95% por bootstrap (`eda.bootstrap_median_lift`), só quando há
  evidência (IC não cruza 0) num nível de contexto com `n >= 20` nos dois
  grupos (fallback `platform x category` -> `platform` -> `global`, exceto
  para `format`/`music_voice_ratio`, que exigem evidência NA PLATAFORMA —
  sem fallback para `global`). Resultados de bootstrap ficam em cache.

Atributos acionáveis (CLAUDE.md, regra d): `video_duration_s`, `format`,
`has_subtitle`, `has_cta`, `has_hook`, `has_face`, `text_density`,
`music_voice_ratio`. `platform`/`category`/`objective`/`target_audience_age`/
`is_retargeting` só condicionam a recomendação, nunca são recomendados.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from klike import eda
from klike import features as feat
from klike.data import load_clean
from klike.model import MODEL_PATH, train_and_save

METRICAS_PERFORMANCE = ["ctr", "engagement_rate", "watch_rate"]
N_MIN_FALLBACK = 20
LIMIAR_SCORE_DESPREZIVEL = 0.5

_df_cache: pd.DataFrame | None = None
_pipeline_cache = None
_cache_lift: dict = {}


def _carregar_dados() -> pd.DataFrame:
    """`load_clean()` + faixas/métricas derivadas da Parte 1, com cache em processo."""
    global _df_cache
    if _df_cache is None:
        df = load_clean()
        df = eda.discretize_duration(df)
        df = eda.discretize_music_voice(df)
        df = eda.add_derived_metrics(df)
        _df_cache = df
    return _df_cache


def _carregar_modelo():
    """Carrega `models/klike_model.joblib`; se não existir, treina e salva (Parte 2)."""
    global _pipeline_cache
    if _pipeline_cache is None:
        if not MODEL_PATH.exists():
            from sklearn.ensemble import HistGradientBoostingRegressor

            modelo = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.1, random_state=42)
            _pipeline_cache = train_and_save(modelo)
        else:
            _pipeline_cache = joblib.load(MODEL_PATH)
    return _pipeline_cache


def _campanha_para_X(campanha) -> pd.DataFrame:
    if isinstance(campanha, dict):
        campanha = pd.Series(campanha)
    linha = pd.DataFrame([campanha])
    return feat.montar_X_preveiculacao(linha)


def _prever_score(campanha) -> float:
    pipeline = _carregar_modelo()
    X = _campanha_para_X(campanha)
    return float(pipeline.predict(X)[0])


def _lift_com_fallback(
    df: pd.DataFrame,
    acao_id: str,
    mascara_referencia: pd.Series,
    mascara_grupo: pd.Series,
    metrica: str,
    campanha,
    permitir_global: bool = True,
) -> dict | None:
    """Lift bootstrap de `metrica` (grupo vs. referência), com fallback de
    contexto `platform x category` -> `platform` -> (`global`, se
    `permitir_global`), exigindo `n >= 20` nos dois grupos em cada nível.
    Cacheado por (acao_id, metrica, nível, contexto). Retorna `None` se nenhum
    nível tiver amostra suficiente."""
    plataforma = campanha["platform"]
    categoria = campanha["category"]

    niveis = [
        ("platform_x_category", (df["platform"] == plataforma) & (df["category"] == categoria)),
        ("platform", df["platform"] == plataforma),
    ]
    if permitir_global:
        niveis.append(("global", pd.Series(True, index=df.index)))

    for nome_nivel, filtro in niveis:
        valores_ref = df.loc[filtro & mascara_referencia, metrica].dropna()
        valores_grp = df.loc[filtro & mascara_grupo, metrica].dropna()
        if len(valores_ref) < N_MIN_FALLBACK or len(valores_grp) < N_MIN_FALLBACK:
            continue

        chave_cache = (acao_id, metrica, nome_nivel, plataforma if nome_nivel != "global" else None, categoria if nome_nivel == "platform_x_category" else None)
        if chave_cache not in _cache_lift:
            _cache_lift[chave_cache] = eda.bootstrap_median_lift(
                valores_ref.to_numpy(), valores_grp.to_numpy(), tipo_efeito="relativo"
            )
        resultado = dict(_cache_lift[chave_cache])
        resultado["nivel_fallback"] = nome_nivel
        resultado["metrica"] = metrica
        return resultado

    return None


def _gerar_candidatos(campanha) -> list[dict]:
    """Para a `campanha`, retorna as ações candidatas (só as que ela ainda não
    segue), cada uma com: `id`, `atributo`, `texto_acao`, `valor_atual`,
    `valor_sugerido` (texto), `valor_whatif` (para o modelo), máscaras de
    referência/grupo (para o lift) e `exige_evidencia_performance`.
    """
    df = _carregar_dados()
    candidatos = []

    if not bool(campanha["has_hook"]):
        candidatos.append(
            {
                "id": "has_hook",
                "atributo": "has_hook",
                "texto_acao": "Adicionar um hook nos primeiros 3 segundos",
                "valor_atual": False,
                "valor_sugerido_texto": "com hook",
                "valor_whatif": True,
                "mascara_referencia": df["has_hook"] == False,  # noqa: E712
                "mascara_grupo": df["has_hook"] == True,  # noqa: E712
                "exige_evidencia_performance": False,
            }
        )

    if not bool(campanha["has_face"]):
        candidatos.append(
            {
                "id": "has_face",
                "atributo": "has_face",
                "texto_acao": "Incluir rosto humano no vídeo",
                "valor_atual": False,
                "valor_sugerido_texto": "com rosto humano",
                "valor_whatif": True,
                "mascara_referencia": df["has_face"] == False,  # noqa: E712
                "mascara_grupo": df["has_face"] == True,  # noqa: E712
                "exige_evidencia_performance": False,
            }
        )

    subtitle_atual = campanha["has_subtitle"]
    if pd.isna(subtitle_atual):
        candidatos.append(
            {
                "id": "has_subtitle",
                "atributo": "has_subtitle",
                "texto_acao": "Verificar se o vídeo tem legendas (não informado nesta campanha) e adicionar, se não tiver",
                "valor_atual": None,
                "valor_sugerido_texto": "com legendas",
                "valor_whatif": True,
                "mascara_referencia": df["has_subtitle"] == False,  # noqa: E712
                "mascara_grupo": df["has_subtitle"] == True,  # noqa: E712
                "exige_evidencia_performance": False,
            }
        )
    elif not bool(subtitle_atual):
        candidatos.append(
            {
                "id": "has_subtitle",
                "atributo": "has_subtitle",
                "texto_acao": "Adicionar legendas",
                "valor_atual": False,
                "valor_sugerido_texto": "com legendas",
                "valor_whatif": True,
                "mascara_referencia": df["has_subtitle"] == False,  # noqa: E712
                "mascara_grupo": df["has_subtitle"] == True,  # noqa: E712
                "exige_evidencia_performance": False,
            }
        )

    if not bool(campanha["has_cta"]):
        candidatos.append(
            {
                "id": "has_cta",
                "atributo": "has_cta",
                "texto_acao": "Adicionar uma chamada para ação (CTA)",
                "valor_atual": False,
                "valor_sugerido_texto": "com CTA",
                "valor_whatif": True,
                "mascara_referencia": df["has_cta"] == False,  # noqa: E712
                "mascara_grupo": df["has_cta"] == True,  # noqa: E712
                "exige_evidencia_performance": False,
            }
        )

    texto_density_atual = campanha["text_density"]
    if texto_density_atual == "high":
        candidatos.append(
            {
                "id": "text_density_high_medium",
                "atributo": "text_density",
                "texto_acao": "Reduzir a quantidade de texto na tela de muito para médio",
                "valor_atual": "high",
                "valor_sugerido_texto": "medium",
                "valor_whatif": "medium",
                "mascara_referencia": df["text_density"] == "high",
                "mascara_grupo": df["text_density"] == "medium",
                "exige_evidencia_performance": False,
            }
        )
    elif texto_density_atual == "medium":
        candidatos.append(
            {
                "id": "text_density_medium_low",
                "atributo": "text_density",
                "texto_acao": "Reduzir a quantidade de texto na tela de médio para pouco",
                "valor_atual": "medium",
                "valor_sugerido_texto": "low",
                "valor_whatif": "low",
                "mascara_referencia": df["text_density"] == "medium",
                "mascara_grupo": df["text_density"] == "low",
                "exige_evidencia_performance": False,
            }
        )

    duracao_atual = campanha["video_duration_s"]
    plataforma = campanha["platform"]
    if pd.notna(duracao_atual):
        if duracao_atual > 30:
            alvo_mediana = df.loc[(df["platform"] == plataforma) & (df["video_duration_bin"] == "15-30s"), "video_duration_s"].median()
            candidatos.append(
                {
                    "id": "duracao_mais30_para_30",
                    "atributo": "video_duration_s",
                    "texto_acao": f"Reduzir a duração do vídeo para até 30s (mediana na plataforma: ~{alvo_mediana:.0f}s)",
                    "valor_atual": duracao_atual,
                    "valor_sugerido_texto": f"~{alvo_mediana:.0f}s",
                    "valor_whatif": float(alvo_mediana),
                    "mascara_referencia": df["video_duration_bin"] == ">30s",
                    "mascara_grupo": df["video_duration_bin"] == "15-30s",
                    "exige_evidencia_performance": False,
                }
            )
        elif duracao_atual > 15:
            alvo_mediana = df.loc[(df["platform"] == plataforma) & (df["video_duration_bin"] == "<=15s"), "video_duration_s"].median()
            candidatos.append(
                {
                    "id": "duracao_15a30_para_15",
                    "atributo": "video_duration_s",
                    "texto_acao": f"Reduzir a duração do vídeo para até 15s (mediana na plataforma: ~{alvo_mediana:.0f}s)",
                    "valor_atual": duracao_atual,
                    "valor_sugerido_texto": f"~{alvo_mediana:.0f}s",
                    "valor_whatif": float(alvo_mediana),
                    "mascara_referencia": df["video_duration_bin"] == "15-30s",
                    "mascara_grupo": df["video_duration_bin"] == "<=15s",
                    "exige_evidencia_performance": False,
                }
            )

    formato_atual = campanha["format"]
    if formato_atual != "vertical":
        candidatos.append(
            {
                "id": f"format_{formato_atual}_vertical",
                "atributo": "format",
                "texto_acao": "Trocar o formato do vídeo para vertical",
                "valor_atual": formato_atual,
                "valor_sugerido_texto": "vertical",
                "valor_whatif": "vertical",
                "mascara_referencia": df["format"] == formato_atual,
                "mascara_grupo": df["format"] == "vertical",
                "exige_evidencia_performance": True,
                "permitir_fallback_global": False,
            }
        )

    music_bin_atual = campanha.get("music_voice_bin") if hasattr(campanha, "get") else None
    if music_bin_atual is None and pd.notna(campanha["music_voice_ratio"]):
        music_bin_atual = "mais_voz" if campanha["music_voice_ratio"] <= 0.33 else ("equilibrado" if campanha["music_voice_ratio"] <= 0.66 else "mais_musica")
    if music_bin_atual == "mais_voz":
        alvo_mediana = df.loc[(df["platform"] == plataforma) & (df["music_voice_bin"] == "equilibrado"), "music_voice_ratio"].median()
        candidatos.append(
            {
                "id": "music_voice_mais_voz_equilibrado",
                "atributo": "music_voice_ratio",
                "texto_acao": f"Ajustar a proporção música/voz para mais equilibrada (~{alvo_mediana:.2f})",
                "valor_atual": campanha["music_voice_ratio"],
                "valor_sugerido_texto": f"~{alvo_mediana:.2f}",
                "valor_whatif": float(alvo_mediana),
                "mascara_referencia": df["music_voice_bin"] == "mais_voz",
                "mascara_grupo": df["music_voice_bin"] == "equilibrado",
                "exige_evidencia_performance": True,
                "permitir_fallback_global": False,
            }
        )

    return candidatos


def _formatar_percentual(valor: float) -> str:
    sinal = "+" if valor >= 0 else ""
    return f"{sinal}{valor:.0f}%"


def _texto_metrica(nome_metrica: str) -> str:
    return {"ctr": "CTR", "engagement_rate": "engajamento", "watch_rate": "fração assistida do vídeo"}.get(nome_metrica, nome_metrica)


def _avaliar_candidato(campanha, candidato: dict) -> dict | None:
    df = _carregar_dados()

    # 1) impacto no score (what-if)
    campanha_mod = campanha.copy()
    campanha_mod[candidato["atributo"]] = candidato["valor_whatif"]
    score_atual = _prever_score(campanha)
    score_novo = _prever_score(campanha_mod)
    delta_score = score_novo - score_atual

    # 2) impacto em performance (com evidência)
    permitir_global = candidato.get("permitir_fallback_global", True)
    metricas_com_evidencia = []
    for metrica in METRICAS_PERFORMANCE:
        resultado = _lift_com_fallback(
            df, candidato["id"], candidato["mascara_referencia"], candidato["mascara_grupo"], metrica, campanha, permitir_global
        )
        if resultado is not None and resultado["evidencia"]:
            metricas_com_evidencia.append(
                {
                    "metrica": metrica,
                    "lift_pct": resultado["efeito_relatado"],
                    "ic_baixo": resultado["ic_baixo"],
                    "ic_alto": resultado["ic_alto"],
                    "n_referencia": resultado["n_referencia"],
                    "n_grupo": resultado["n_grupo"],
                    "nivel_fallback": resultado["nivel_fallback"],
                }
            )

    if candidato["exige_evidencia_performance"] and not metricas_com_evidencia:
        return None  # format/música-voz: nunca recomendar sem evidência na plataforma

    if not metricas_com_evidencia and delta_score <= LIMIAR_SCORE_DESPREZIVEL:
        return None  # regra 11: sem evidência de performance E score desprezível/negativo -> descarta

    # texto do impacto no score
    if delta_score < 0:
        texto_score = "o modelo não estima ganho de score para esta campanha específica"
    elif delta_score <= LIMIAR_SCORE_DESPREZIVEL:
        texto_score = "impacto no score: desprezível segundo o modelo"
    else:
        texto_score = f"O modelo estima {delta_score:+.0f} pontos no Klike Score."

    # texto do impacto em performance
    partes_performance = []
    for m in metricas_com_evidencia:
        nome_nivel = {"platform_x_category": "nesta plataforma e categoria", "platform": f"na {campanha['platform']}", "global": "no geral"}[m["nivel_fallback"]]
        partes_performance.append(
            f"{nome_nivel}, criativos nessa condição têm {_texto_metrica(m['metrica'])} mediano {_formatar_percentual(m['lift_pct'])} "
            f"(faixa de incerteza: {_formatar_percentual(m['ic_baixo'])} a {_formatar_percentual(m['ic_alto'])}; n={m['n_grupo']} vs {m['n_referencia']})"
        )

    if partes_performance:
        texto = f"{candidato['texto_acao']}: " + "; ".join(partes_performance) + ". " + texto_score
    else:
        texto = f"{candidato['texto_acao']}: sem evidência de impacto em performance nos dados; impacto estimado apenas no score. " + texto_score

    nivel_prioridade = 1 if metricas_com_evidencia else 2
    if metricas_com_evidencia:
        lifts_evidencia = {m["metrica"]: m["lift_pct"] for m in metricas_com_evidencia}
        ordenacao = lifts_evidencia.get("ctr", max(lifts_evidencia.values()))
    else:
        ordenacao = delta_score

    return {
        "acao": candidato["texto_acao"],
        "atributo": candidato["atributo"],
        "valor_atual": candidato["valor_atual"],
        "valor_sugerido": candidato["valor_sugerido_texto"],
        "delta_score": None if delta_score <= LIMIAR_SCORE_DESPREZIVEL and metricas_com_evidencia else round(delta_score, 1),
        "metricas_com_evidencia": metricas_com_evidencia,
        "nivel_prioridade": nivel_prioridade,
        "texto": texto,
        "_ordenacao": ordenacao,
    }


def carregar_dados() -> pd.DataFrame:
    """Dataset limpo + faixas/métricas derivadas da Parte 1 (cacheado). Usado
    pela CLI para buscar uma campanha por `campaign_id`."""
    return _carregar_dados()


def prever_score(campanha) -> float:
    """Score Klike previsto pelo modelo salvo (Parte 2) para `campanha`."""
    return _prever_score(campanha)


def recomendar(campanha, n_min: int = 3) -> list[dict]:
    """Gera recomendações acionáveis e quantificadas para `campanha` (`pd.Series`
    ou `dict` com as colunas do dataset), ordenadas por prioridade (nível 1:
    evidência em performance, pelo maior lift; nível 2: só impacto no score,
    pelo maior delta). Retorna pelo menos `n_min` recomendações quando há ações
    aplicáveis; se a campanha já segue quase tudo, retorna uma única entrada
    explicando isso em vez de inventar recomendação.
    """
    if isinstance(campanha, dict):
        campanha = pd.Series(campanha)

    candidatos = _gerar_candidatos(campanha)
    avaliados = [r for c in candidatos if (r := _avaliar_candidato(campanha, c)) is not None]

    avaliados.sort(key=lambda r: (r["nivel_prioridade"], -r["_ordenacao"]))
    for r in avaliados:
        del r["_ordenacao"]

    if not avaliados:
        return [
            {
                "acao": None,
                "atributo": None,
                "valor_atual": None,
                "valor_sugerido": None,
                "delta_score": None,
                "metricas_com_evidencia": [],
                "nivel_prioridade": 3,
                "texto": "A campanha já segue as boas práticas conhecidas (ou não há ações com evidência suficiente). Nenhuma recomendação adicional.",
            }
        ]

    if len(avaliados) < n_min:
        avaliados.append(
            {
                "acao": None,
                "atributo": None,
                "valor_atual": None,
                "valor_sugerido": None,
                "delta_score": None,
                "metricas_com_evidencia": [],
                "nivel_prioridade": 3,
                "texto": (
                    f"A campanha já segue a maior parte das boas práticas conhecidas — só {len(avaliados)} "
                    "recomendação(ões) com evidência suficiente, abaixo do mínimo pedido. Não há mais ações a sugerir."
                ),
            }
        )

    return avaliados


def distribuicao_delta_score(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Diagnóstico (seção de sanidade do notebook 03): para cada tipo de ação,
    o `delta_score` (what-if) em TODAS as campanhas onde ela é candidata —
    mesmo as que seriam descartadas das recomendações finais (regra 11,
    score desprezível/negativo sem evidência) — para mostrar o quanto o
    efeito no score varia entre campanhas. Mediana, mínimo e máximo por ação.
    """
    if df is None:
        df = _carregar_dados()

    linhas = []
    for _, campanha in df.iterrows():
        for candidato in _gerar_candidatos(campanha):
            campanha_mod = campanha.copy()
            campanha_mod[candidato["atributo"]] = candidato["valor_whatif"]
            delta = _prever_score(campanha_mod) - _prever_score(campanha)
            linhas.append({"acao": candidato["texto_acao"].split(" (")[0], "delta_score": delta})

    tabela = pd.DataFrame(linhas)
    return tabela.groupby("acao")["delta_score"].agg(n="count", mediana="median", minimo="min", maximo="max").sort_values("mediana", ascending=False).round(2)
