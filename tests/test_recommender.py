"""Testes rápidos do Recommendations Engine (Parte 3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from klike import recommender as rec  # noqa: E402


def _campanha_ruim():
    """Campanha sintética sem hook, sem rosto, com muito texto — deve gerar
    várias recomendações aplicáveis."""
    df = rec.carregar_dados()
    base = df.iloc[0].copy()
    base["platform"] = "Meta"
    base["category"] = "E-commerce"
    base["format"] = "horizontal"
    base["has_hook"] = False
    base["has_face"] = False
    base["has_cta"] = False
    base["has_subtitle"] = False
    base["text_density"] = "high"
    base["video_duration_s"] = 45.0
    base["music_voice_ratio"] = 0.1
    return base


def test_nunca_recomenda_pratica_ja_seguida():
    """Uma campanha que já segue TODAS as práticas acionáveis não deve
    receber recomendação de nenhuma dessas práticas."""
    df = rec.carregar_dados()
    base = df.iloc[0].copy()
    base["has_hook"] = True
    base["has_face"] = True
    base["has_cta"] = True
    base["has_subtitle"] = True
    base["text_density"] = "low"
    base["video_duration_s"] = 10.0
    base["format"] = "vertical"

    recomendacoes = rec.recomendar(base, n_min=1)
    atributos_recomendados = {r["atributo"] for r in recomendacoes if r["atributo"] is not None}

    assert "has_hook" not in atributos_recomendados
    assert "has_face" not in atributos_recomendados
    assert "has_cta" not in atributos_recomendados
    assert "has_subtitle" not in atributos_recomendados
    assert "text_density" not in atributos_recomendados
    assert "video_duration_s" not in atributos_recomendados
    assert "format" not in atributos_recomendados


def test_nunca_recomenda_formato_sem_dados_no_tiktok():
    """`format=horizontal` não existe no TikTok (0 campanhas) — mesmo que a
    campanha sintética force esse valor, não deve haver recomendação de
    formato sem evidência no contexto."""
    df = rec.carregar_dados()
    base = df.iloc[0].copy()
    base["platform"] = "TikTok"
    base["format"] = "horizontal"

    recomendacoes = rec.recomendar(base, n_min=1)
    assert all(r["atributo"] != "format" for r in recomendacoes)


def test_ordem_respeita_niveis_de_prioridade():
    """As recomendações com evidência em performance (nível 1) devem vir
    antes das que só têm impacto no score (nível 2), que vêm antes de
    qualquer nota informativa (nível 3)."""
    recomendacoes = rec.recomendar(_campanha_ruim(), n_min=3)
    niveis = [r["nivel_prioridade"] for r in recomendacoes]
    assert niveis == sorted(niveis)


def test_campanha_ruim_recebe_pelo_menos_tres_recomendacoes():
    recomendacoes = rec.recomendar(_campanha_ruim(), n_min=3)
    reais = [r for r in recomendacoes if r["acao"] is not None]
    assert len(reais) >= 3


def test_fallback_sobe_de_nivel_sem_evidencia_mesmo_com_n_suficiente():
    """KLK-0024 (Meta/Lead Gen): no nível `platform x category`, o CTR de
    `has_hook` tem n >= 20 nos dois grupos mas o IC cruza 0 (sem evidência).
    O fallback deve subir para o nível `platform`, onde há evidência
    (CTR +57% na Meta) — não pode ficar sem evidência só porque o nível mais
    específico tinha amostra suficiente."""
    df = rec.carregar_dados()
    campanha = df.loc[df["campaign_id"] == "KLK-0024"].iloc[0]
    recomendacoes = rec.recomendar(campanha, n_min=3)

    hook = next(r for r in recomendacoes if r["atributo"] == "has_hook")
    metricas = {m["metrica"]: m for m in hook["metricas_com_evidencia"]}
    assert "ctr" in metricas
    assert metricas["ctr"]["nivel_fallback"] == "platform"
    assert "na Meta em geral" in hook["texto"]


def test_prioridade_com_evidencia_em_ctr_vem_antes_das_sem_ctr():
    """Dentro do nível 1 (evidência em performance), ações com evidência em
    CTR devem vir antes das que só têm evidência em outras métricas —
    independente da magnitude do lift."""
    df = rec.carregar_dados()
    campanha = df.loc[df["campaign_id"] == "KLK-0024"].iloc[0]
    recomendacoes = rec.recomendar(campanha, n_min=3)

    nivel1 = [r for r in recomendacoes if r["nivel_prioridade"] == 1]
    tem_ctr = ["ctr" in {m["metrica"] for m in r["metricas_com_evidencia"]} for r in nivel1]

    # nenhuma ação sem evidência em CTR pode vir antes de uma com evidência em CTR
    assert tem_ctr == sorted(tem_ctr, reverse=True)
    # e, dentro de quem tem CTR, a ordem é pelo lift de CTR (maior primeiro)
    lifts_ctr = [
        next(m["lift_pct"] for m in r["metricas_com_evidencia"] if m["metrica"] == "ctr")
        for r, c in zip(nivel1, tem_ctr)
        if c
    ]
    assert lifts_ctr == sorted(lifts_ctr, reverse=True)


def test_rosto_usa_nivel_plataforma_e_nao_ctr_global():
    """KLK-0024 (Meta): `has_face` não tem evidência de CTR na Meta (IC cruza
    0, n suficiente) — só em engajamento e fração assistida. Por isso a
    recomendação deve usar o nível `platform` (Meta) e reportar essas duas
    métricas, SEM subir para o nível `global` em busca de CTR (que só tem
    evidência para rosto humano no TikTok, não na Meta — Parte 1; usar o CTR
    global contradiria esse achado)."""
    df = rec.carregar_dados()
    campanha = df.loc[df["campaign_id"] == "KLK-0024"].iloc[0]
    recomendacoes = rec.recomendar(campanha, n_min=3)

    rosto = next(r for r in recomendacoes if r["atributo"] == "has_face")
    metricas = {m["metrica"]: m for m in rosto["metricas_com_evidencia"]}

    assert "ctr" not in metricas
    assert "engagement_rate" in metricas
    assert metricas["engagement_rate"]["nivel_fallback"] == "platform"
    assert "watch_rate" in metricas
    assert metricas["watch_rate"]["nivel_fallback"] == "platform"
    assert "na Meta em geral" in rosto["texto"]
    assert "no geral" not in rosto["texto"]
