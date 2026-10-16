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
