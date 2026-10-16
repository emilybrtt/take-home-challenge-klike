"""Recommendations Engine (Parte 3): recomendações acionáveis e quantificadas
para uma campanha, combinando o modelo preditivo (Parte 2) e os achados de
performance da EDA (Parte 1).

Duas fontes de quantificação, sempre explícitas (CLAUDE.md, regra e):
- **Impacto no score**: "what-if" com o modelo salvo — sempre disponível.
- **Impacto em performance** (ctr/engagement_rate/watch_rate): lift da mediana
  com IC95% por bootstrap (`eda.bootstrap_median_lift`), decidido para a AÇÃO
  como um todo (`_lift_fallback_para_acao`), não métrica a métrica: percorre
  os níveis de contexto `platform x category` -> `platform` -> `global`
  (exceto para `format`/`music_voice_ratio`, que exigem evidência NA
  PLATAFORMA — sem fallback para `global`) e PARA no primeiro nível com
  evidência (IC não cruza 0) em QUALQUER métrica, reportando só as métricas
  com evidência NESSE nível. Sobe de nível por falta de amostra (`n < 20`)
  em qualquer transição, mas por falta de evidência (amostra suficiente, sem
  efeito) só sobe de `platform x category` para `platform` — nunca de
  `platform` para `global`: se a plataforma da campanha já tem dado
  suficiente e não mostra efeito, essa é a resposta contextual correta, e
  importar o efeito de outras plataformas poderia contradizer um achado
  específico da plataforma (ex.: rosto humano só tem evidência de CTR no
  TikTok — Parte 1). O texto da recomendação diz explicitamente o nível
  usado (ex.: "na Meta em geral, ..."). Resultados de bootstrap ficam em
  cache.

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


def _lift_fallback_para_acao(
    df: pd.DataFrame,
    acao_id: str,
    mascara_referencia: pd.Series,
    mascara_grupo: pd.Series,
    campanha,
    permitir_global: bool = True,
) -> dict[str, dict]:
    """Lift bootstrap das 3 métricas de performance para UMA ação, com
    fallback de contexto `platform x category` -> `platform` -> (`global`, se
    `permitir_global`), decidido para a ação como um todo (não métrica a
    métrica): percorre os níveis e PARA no primeiro que tiver evidência (IC
    não cruza 0) em QUALQUER métrica, retornando só as métricas com evidência
    NESSE nível — nunca busca uma métrica sem evidência (ex.: CTR) em outro
    nível depois que a ação já "resolveu" num nível mais específico.

    A escalada de nível tem dois motivos possíveis, mas só um é válido entre
    `platform` e `global`:
    - **falta de amostra** (`n < 20` nos dois grupos, em TODAS as métricas
      do nível) — sempre sobe, em qualquer transição;
    - **falta de evidência** (amostra suficiente, mas nenhuma métrica com IC
      fora de 0) — só sobe de `platform x category` para `platform` (ainda é
      a mesma plataforma, só perde a especificidade de categoria). De
      `platform` para `global` NUNCA sobe por esse motivo: se a plataforma
      tem dado suficiente e não mostra efeito, essa É a resposta contextual
      correta — misturar com outras plataformas poderia contradizer um
      achado específico da plataforma da campanha (ex.: rosto humano só tem
      evidência de CTR no TikTok — Parte 1 — então não se usa o CTR do
      "geral" para justificar rosto humano na Meta).

    Cacheado por (acao_id, metrica, nível, contexto). Retorna `{}` se nenhum
    nível tiver evidência (ou amostra) para nenhuma métrica."""
    plataforma = campanha["platform"]
    categoria = campanha["category"]

    niveis = [
        ("platform_x_category", (df["platform"] == plataforma) & (df["category"] == categoria)),
        ("platform", df["platform"] == plataforma),
    ]
    if permitir_global:
        niveis.append(("global", pd.Series(True, index=df.index)))

    for i, (nome_nivel, filtro) in enumerate(niveis):
        resultados_nivel = {}
        for metrica in METRICAS_PERFORMANCE:
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
            resultados_nivel[metrica] = resultado

        if not resultados_nivel:
            continue  # amostra insuficiente em TODAS as métricas deste nível -> sobe

        com_evidencia = {m: r for m, r in resultados_nivel.items() if r["evidencia"]}
        if com_evidencia:
            return com_evidencia  # para aqui: só as métricas com evidência deste nível

        proximo_nivel_e_global = i + 1 < len(niveis) and niveis[i + 1][0] == "global"
        if proximo_nivel_e_global:
            return {}  # amostra suficiente, sem evidência — não sobe de platform para global

    return {}


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
                "descricao_referencia": "sem hook",
                "descricao_grupo": "com hook",
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
                "descricao_referencia": "sem rosto humano",
                "descricao_grupo": "com rosto humano",
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
                "descricao_referencia": "sem legendas",
                "descricao_grupo": "com legendas",
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
                "descricao_referencia": "sem legendas",
                "descricao_grupo": "com legendas",
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
                "descricao_referencia": "sem CTA",
                "descricao_grupo": "com CTA",
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
                "descricao_referencia": "com muito texto na tela",
                "descricao_grupo": "com texto médio na tela",
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
                "descricao_referencia": "com texto médio na tela",
                "descricao_grupo": "com pouco texto na tela",
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
                    "descricao_referencia": "com vídeo de mais de 30s",
                    "descricao_grupo": "com vídeo de 15 a 30s",
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
                    "descricao_referencia": "com vídeo de 15 a 30s",
                    "descricao_grupo": "com vídeo de até 15s",
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
                "descricao_referencia": f"em formato {formato_atual}",
                "descricao_grupo": "em formato vertical",
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
                "descricao_referencia": "com predominância de voz",
                "descricao_grupo": "com proporção música/voz mais equilibrada",
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

    # 2) impacto em performance (com evidência) — decidido para a ação como um
    # todo: um único nível de contexto, com todas as métricas com evidência
    # nesse nível (ver `_lift_fallback_para_acao`)
    permitir_global = candidato.get("permitir_fallback_global", True)
    resultados_evidencia = _lift_fallback_para_acao(
        df, candidato["id"], candidato["mascara_referencia"], candidato["mascara_grupo"], campanha, permitir_global
    )
    metricas_com_evidencia = [
        {
            "metrica": metrica,
            "lift_pct": resultados_evidencia[metrica]["efeito_relatado"],
            "ic_baixo": resultados_evidencia[metrica]["ic_baixo"],
            "ic_alto": resultados_evidencia[metrica]["ic_alto"],
            "n_referencia": resultados_evidencia[metrica]["n_referencia"],
            "n_grupo": resultados_evidencia[metrica]["n_grupo"],
            "nivel_fallback": resultados_evidencia[metrica]["nivel_fallback"],
        }
        for metrica in METRICAS_PERFORMANCE
        if metrica in resultados_evidencia
    ]

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

    # texto do impacto em performance — todas as métricas com evidência vêm do
    # MESMO nível de contexto (ver `_lift_fallback_para_acao`), então o nível
    # é dito uma única vez, não repetido por métrica
    if metricas_com_evidencia:
        nome_nivel = {
            "platform_x_category": "nesta plataforma e categoria",
            "platform": f"na {campanha['platform']} em geral",
            "global": "no geral",
        }[metricas_com_evidencia[0]["nivel_fallback"]]
        clausulas_metricas = []
        for m in metricas_com_evidencia:
            comparativo = "maior" if m["lift_pct"] >= 0 else "menor"
            clausulas_metricas.append(
                f"{_texto_metrica(m['metrica'])} mediano {abs(m['lift_pct']):.0f}% {comparativo} que os {candidato['descricao_referencia']} "
                f"(faixa de incerteza: {_formatar_percentual(m['ic_baixo'])} a {_formatar_percentual(m['ic_alto'])}; n={m['n_grupo']} vs {m['n_referencia']})"
            )
        texto_performance = f"{nome_nivel}, criativos {candidato['descricao_grupo']} têm " + "; ".join(clausulas_metricas)
        texto = f"{candidato['texto_acao']}: {texto_performance}. " + texto_score
    else:
        texto = f"{candidato['texto_acao']}: sem evidência de impacto em performance nos dados; impacto estimado apenas no score. " + texto_score

    nivel_prioridade = 1 if metricas_com_evidencia else 2
    if metricas_com_evidencia:
        lifts_evidencia = {m["metrica"]: m["lift_pct"] for m in metricas_com_evidencia}
        tem_ctr = "ctr" in lifts_evidencia
        ordenacao = lifts_evidencia["ctr"] if tem_ctr else max(lifts_evidencia.values())
    else:
        tem_ctr = False
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
        "_tem_ctr": tem_ctr,
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

    avaliados.sort(key=lambda r: (r["nivel_prioridade"], 0 if r["_tem_ctr"] else 1, -r["_ordenacao"]))
    for r in avaliados:
        del r["_ordenacao"]
        del r["_tem_ctr"]

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
