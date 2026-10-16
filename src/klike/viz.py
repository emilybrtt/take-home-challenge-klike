"""Visualizações finais da Parte 1, para um público não técnico (time de marketing).

Convenções de todas as funções deste módulo:
- Título = o insight em uma frase (não a descrição do gráfico).
- Rótulos em português, sem nomes de coluna.
- `n` de cada grupo mostrado de forma discreta (tamanho de fonte pequeno, cor secundária).
- Paleta de plataforma consistente: Meta = azul, TikTok = laranja, LinkedIn = água
  (validada com `scripts/validate_palette.js` do skill dataviz: CVD-safe).
- Nota de rodapé fixa: "Associação observada em 500 campanhas; não implica causalidade."
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

NOTA_RODAPE = "Associação observada em 500 campanhas; não implica causalidade."

COR_PLATAFORMA = {"Meta": "#2a78d6", "TikTok": "#eb6834", "LinkedIn": "#1baf7a"}
COR_ROBUSTO = "#2a78d6"
COR_ISOLADO = "#898781"
COR_SEM_EVIDENCIA = "#c3c2b7"
COR_DIVERGENTE = LinearSegmentedColormap.from_list("divergente", ["#d03b3b", "#f0efec", "#2a78d6"])
RAMPA_SEQUENCIAL_AZUL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

INK_PRIMARIO = "#0b0b0b"
INK_SECUNDARIO = "#52514e"
INK_MUTED = "#898781"
GRADE = "#e1e0d9"

RUBRICA_LABELS = {
    ("has_hook", True): "Hook nos 3 primeiros segundos",
    ("has_face", True): "Rosto humano no vídeo",
    ("has_cta", True): "Call-to-action (CTA)",
    ("has_subtitle", True): "Legendas",
    ("text_density", "medium"): "Texto médio na tela (vs. pouco)",
    ("text_density", "high"): "Muito texto na tela (vs. pouco)",
    ("format", "horizontal"): "Formato horizontal (vs. vertical)",
    ("format", "quadrado"): "Formato quadrado (vs. vertical)",
    ("video_duration_bin", ">30s"): "Vídeo com mais de 30s (vs. até 15s)",
    ("video_duration_bin", "15-30s"): "Vídeo de 15 a 30s (vs. até 15s)",
    ("music_voice_bin", "equilibrado"): "Música e voz equilibradas (vs. mais voz)",
    ("music_voice_bin", "mais_musica"): "Predomínio de música (vs. mais voz)",
}


def _pt(texto_formatado: str) -> str:
    """Troca o separador decimal de ponto para vírgula (pt-BR): '3.9x' -> '3,9x'."""
    return texto_formatado.replace(".", ",")


def _rodape(fig):
    fig.text(0.01, 0.01, NOTA_RODAPE, fontsize=8, color=INK_MUTED, ha="left")


def _estilo_eixo(ax):
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECUNDARIO)
    ax.grid(axis="x", color=GRADE, linewidth=0.8, zorder=0)


def forest_plot_ctr_engajamento_score(tabela_geral: pd.DataFrame, achados_classificados: pd.DataFrame):
    """Gráfico 1: forest plot em 3 painéis (CTR %, engajamento %, score em pontos).

    `tabela_geral`: saída de `eda.effect_by_attribute` concatenada (análise geral,
    sem coluna `platform`), para os atributos/níveis com evidência em CTR,
    engajamento OU score. `achados_classificados`: saída de `eda.classificar_robustez`
    (para pintar robusto vs. isolado).
    """
    sub_painel = tabela_geral[(~tabela_geral["e_referencia"]) & tabela_geral["metrica"].isin(["ctr", "engagement_rate", "klike_score"])]
    linhas = sorted({(r["atributo"], r["nivel"]) for _, r in sub_painel[sub_painel["evidencia"]].iterrows()})
    ordem_classificacao = {
        (row["atributo"], row["nivel"]): row["classificacao"] for _, row in achados_classificados.iterrows()
    }

    # ordena pelo efeito no ctr
    efeito_ctr = {
        (r["atributo"], r["nivel"]): r["efeito"]
        for _, r in tabela_geral[(tabela_geral["metrica"] == "ctr") & (~tabela_geral["e_referencia"])].iterrows()
    }
    linhas = sorted(linhas, key=lambda k: efeito_ctr.get(k, 0), reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    titulos = {
        "ctr": "CTR — cliques por impressão",
        "engagement_rate": "Engajamento",
        "klike_score": "Score Klike (pontos)",
    }

    y = np.arange(len(linhas))[::-1]
    for ax, metrica in zip(axes, ["ctr", "engagement_rate", "klike_score"]):
        sub = tabela_geral[(tabela_geral["metrica"] == metrica) & (~tabela_geral["e_referencia"])]
        sub = sub.set_index(["atributo", "nivel"])

        for yi, chave in zip(y, linhas):
            if chave not in sub.index:
                continue
            linha = sub.loc[chave]
            n = int(linha["n"])
            efeito, ic_b, ic_a = linha["efeito"], linha["ic_baixo"], linha["ic_alto"]
            classificacao = ordem_classificacao.get(chave)
            tem_evidencia = bool(linha["evidencia"])

            if not tem_evidencia:
                cor, estilo_linha, alpha = COR_SEM_EVIDENCIA, "--", 0.9
                marcador = "o"
                facecolor = "white"
            elif classificacao == "robusto":
                cor, estilo_linha, alpha = COR_ROBUSTO, "-", 1.0
                marcador = "o"
                facecolor = cor
            else:
                cor, estilo_linha, alpha = COR_ISOLADO, "-", 1.0
                marcador = "o"
                facecolor = cor

            ax.plot([ic_b, ic_a], [yi, yi], color=cor, linestyle=estilo_linha, linewidth=2, alpha=alpha, zorder=2)
            ax.scatter(
                [efeito], [yi], s=55, marker=marcador, color=facecolor, edgecolor=cor, linewidth=1.3, zorder=3
            )
            sufixo = "pts" if metrica == "klike_score" else "%"
            ax.annotate(
                f"n={n}", (max(ic_b, ic_a), yi), xytext=(6, 0), textcoords="offset points",
                fontsize=7.5, color=INK_MUTED, va="center",
            )

        ax.axvline(0, color=INK_MUTED, linewidth=1, zorder=1)
        ax.set_title(titulos[metrica], fontsize=11, color=INK_SECUNDARIO)
        _estilo_eixo(ax)
        ax.set_xlabel("pontos" if metrica == "klike_score" else "variação % na mediana", fontsize=9, color=INK_SECUNDARIO)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([RUBRICA_LABELS[k] for k in linhas], fontsize=10, color=INK_PRIMARIO)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False)

    legenda = [
        Line2D([0], [0], marker="o", color=COR_ROBUSTO, markerfacecolor=COR_ROBUSTO, linestyle="-", label="Achado robusto"),
        Line2D([0], [0], marker="o", color=COR_ISOLADO, markerfacecolor=COR_ISOLADO, linestyle="-", label="Achado isolado"),
        Line2D([0], [0], marker="o", color=COR_SEM_EVIDENCIA, markerfacecolor="white", linestyle="--", label="Sem evidência nesta métrica"),
    ]
    fig.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Hook e rosto humano são os atributos com efeito mais forte e mais consistente", fontsize=14, y=1.1, color=INK_PRIMARIO)
    _rodape(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def heatmap_ctr_plataforma(tabela_plataforma_ctr: pd.DataFrame, n_minimo: int = 20):
    """Gráfico 2: heatmap atributo x plataforma, para CTR. `tabela_plataforma_ctr`:
    saída de `eda.effect_by_attribute` (métrica ctr) concatenada por plataforma,
    com coluna `platform`. Linhas ausentes para uma combinação atributo-nível x
    plataforma são tratadas como "sem dados" (n=0)."""
    linhas = list(RUBRICA_LABELS.keys())
    plataformas = ["Meta", "TikTok", "LinkedIn"]

    tab = tabela_plataforma_ctr[~tabela_plataforma_ctr["e_referencia"]].set_index(["atributo", "nivel", "platform"])

    fig, ax = plt.subplots(figsize=(8, 7))
    efeitos = []
    for chave in linhas:
        for plataforma in plataformas:
            idx = chave + (plataforma,)
            if idx in tab.index:
                efeitos.append(tab.loc[idx, "efeito"])
    vmax = max(abs(min(efeitos)), abs(max(efeitos)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for yi, chave in enumerate(linhas):
        for xi, plataforma in enumerate(plataformas):
            idx = chave + (plataforma,)
            y_plot = len(linhas) - 1 - yi
            if idx not in tab.index:
                ax.add_patch(Rectangle((xi, y_plot), 1, 1, facecolor="#f0efec", edgecolor="white", linewidth=2))
                ax.text(xi + 0.5, y_plot + 0.5, "sem\ndados", ha="center", va="center", fontsize=8, color=INK_MUTED)
                continue

            linha = tab.loc[idx]
            efeito, n, evidencia = linha["efeito"], int(linha["n"]), bool(linha["evidencia"])
            amostra_insuficiente = n < n_minimo or int(linha["n_referencia"]) < n_minimo

            if amostra_insuficiente:
                ax.add_patch(
                    Rectangle((xi, y_plot), 1, 1, facecolor="#f0efec", edgecolor="white", linewidth=2, hatch="////")
                )
                ax.text(xi + 0.5, y_plot + 0.5, f"n<{n_minimo}", ha="center", va="center", fontsize=8, color=INK_MUTED)
                continue

            cor = COR_DIVERGENTE(norm(efeito)) if evidencia else "#f0efec"
            texto_cor = "white" if evidencia and abs(efeito) > vmax * 0.45 else INK_PRIMARIO
            ax.add_patch(Rectangle((xi, y_plot), 1, 1, facecolor=cor, edgecolor="white", linewidth=2))
            sinal = "+" if efeito > 0 else ""
            ax.text(
                xi + 0.5, y_plot + 0.55, f"{sinal}{efeito:.0f}%", ha="center", va="center",
                fontsize=10, color=texto_cor if evidencia else INK_MUTED, fontweight="bold" if evidencia else "normal",
            )
            ax.text(xi + 0.5, y_plot + 0.22, f"n={n}", ha="center", va="center", fontsize=7, color=texto_cor if evidencia else INK_MUTED, alpha=0.85)

    ax.set_xlim(0, len(plataformas))
    ax.set_ylim(0, len(linhas))
    ax.set_xticks(np.arange(len(plataformas)) + 0.5)
    ax.set_xticklabels(plataformas, fontsize=11, color=INK_PRIMARIO)
    ax.set_yticks(np.arange(len(linhas)) + 0.5)
    ax.set_yticklabels([RUBRICA_LABELS[k] for k in reversed(linhas)], fontsize=10, color=INK_PRIMARIO)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legenda = [
        Patch(facecolor=COR_DIVERGENTE(norm(vmax * 0.8)), label="CTR acima da referência (com evidência)"),
        Patch(facecolor=COR_DIVERGENTE(norm(-vmax * 0.8)), label="CTR abaixo da referência (com evidência)"),
        Patch(facecolor="#f0efec", label="Sem evidência / amostra insuficiente / sem dados"),
    ]
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=1, frameon=False, fontsize=9)

    fig.suptitle("Hook é o único atributo associado a CTR maior nas 3 plataformas", fontsize=13, y=1.04, color=INK_PRIMARIO)
    fig.text(
        0.5, 0.995, "Rosto humano tem evidência só no TikTok; muito texto e vídeo longo, só no Meta",
        ha="center", fontsize=10, color=INK_SECUNDARIO,
    )
    _rodape(fig)
    fig.tight_layout(rect=[0, 0.08, 1, 0.9])
    return fig


def score_x_performance(df: pd.DataFrame, df_roas: pd.DataFrame, n_quintis: int = 5):
    """Gráfico 3: klike_score em quintis x mediana de CTR e de ROAS (subconjunto)."""
    rotulos = [f"Q{i+1}" for i in range(n_quintis)]
    rotulos[0] += "\n(mais baixo)"
    rotulos[-1] += "\n(mais alto)"

    df = df.copy()
    df["quintil"] = pd.qcut(df["klike_score"], n_quintis, labels=rotulos)
    ctr_por_quintil = df.groupby("quintil", observed=True)["ctr"].agg(mediana="median", n="count")

    df_roas = df_roas.copy()
    df_roas["quintil"] = pd.qcut(df_roas["klike_score"], n_quintis, labels=rotulos)
    roas_por_quintil = df_roas.groupby("quintil", observed=True)["roas"].agg(mediana="median", n="count")

    razao = ctr_por_quintil["mediana"].iloc[-1] / ctr_por_quintil["mediana"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, dados, titulo, fmt in [
        (axes[0], ctr_por_quintil, "CTR mediano por faixa de score", "{:.1%}"),
        (axes[1], roas_por_quintil, "ROAS mediano por faixa de score", "{:.2f}x"),
    ]:
        cores = RAMPA_SEQUENCIAL_AZUL[: len(dados)]
        barras = ax.bar(dados.index, dados["mediana"], color=cores, zorder=2, width=0.65)
        for barra, (_, linha) in zip(barras, dados.iterrows()):
            ax.annotate(
                _pt(fmt.format(linha["mediana"])), (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10, color=INK_PRIMARIO, fontweight="bold",
            )
            ax.annotate(
                f"n={int(linha['n'])}", (barra.get_x() + barra.get_width() / 2, 0),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color="white",
            )
        ax.set_title(titulo, fontsize=11, color=INK_SECUNDARIO)
        _estilo_eixo(ax)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=9, colors=INK_PRIMARIO)
        ax.grid(False)

    fig.suptitle(f"Campanhas com score mais alto têm CTR {_pt(f'{razao:.1f}')}x maior", fontsize=14, y=1.02, color=INK_PRIMARIO)
    _rodape(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def checklist_criativo(resumo_faixas: pd.DataFrame):
    """Gráfico 4 ("checklist do criativo"): CTR mediano por nº de boas práticas
    presentes (`n_boas_praticas`, já com a faixa 0-1 unida — n pequeno).

    `resumo_faixas`: DataFrame com colunas `faixa`, `n`, `mediana`, `ic_baixo`,
    `ic_alto` (uma linha por faixa, saída de `eda.bootstrap_median_ci` por grupo).
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cores = RAMPA_SEQUENCIAL_AZUL[: len(resumo_faixas)]

    x = np.arange(len(resumo_faixas))
    for xi, (_, linha), cor in zip(x, resumo_faixas.iterrows(), cores):
        ax.plot([xi, xi], [linha["ic_baixo"], linha["ic_alto"]], color=cor, linewidth=3, zorder=2, solid_capstyle="round")
        ax.scatter([xi], [linha["mediana"]], s=90, color=cor, zorder=3, edgecolor="white", linewidth=1)
        ax.annotate(
            _pt(f"{linha['mediana']:.1%}"), (xi, linha["mediana"]), xytext=(12, 0), textcoords="offset points",
            ha="left", va="center", fontsize=10, color=INK_PRIMARIO, fontweight="bold",
        )
        ax.annotate(
            f"n={int(linha['n'])}", xy=(xi, 0), xycoords=("data", "axes fraction"), xytext=(0, -32),
            textcoords="offset points", ha="center", va="top", fontsize=8, color=INK_MUTED, clip_on=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(resumo_faixas["faixa"], fontsize=10, color=INK_PRIMARIO)
    ax.set_xlabel("nº de boas práticas presentes no criativo", fontsize=9, color=INK_SECUNDARIO, labelpad=28)
    ax.set_ylabel("")
    ax.set_yticks([])
    _estilo_eixo(ax)
    ax.grid(False)
    ax.set_xlim(-0.5, len(resumo_faixas) - 0.3)

    razao = resumo_faixas["mediana"].iloc[-1] / resumo_faixas["mediana"].iloc[0]
    fig.suptitle(
        f"Criativos com as 4 boas práticas têm CTR {_pt(f'{razao:.1f}')}x maior que os com no máximo uma",
        fontsize=13, y=1.0, color=INK_PRIMARIO,
    )
    ax.set_title("Boas práticas: hook, rosto humano, pouco texto na tela, vídeo de até 30s", fontsize=9, color=INK_SECUNDARIO, pad=28)
    _rodape(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


def mix_por_plataforma(df: pd.DataFrame, coluna: str, titulo: str):
    """Gráfico de apoio (fica só no notebook técnico): barras 100% empilhadas
    da composição de `coluna` por `platform`."""
    dados = df[["platform", coluna]].dropna()
    tabela = pd.crosstab(dados["platform"], dados[coluna], normalize="index").mul(100)

    fig, ax = plt.subplots(figsize=(7, 4))
    esquerda = np.zeros(len(tabela))
    cores = RAMPA_SEQUENCIAL_AZUL[: tabela.shape[1]] if tabela.shape[1] <= len(RAMPA_SEQUENCIAL_AZUL) else None
    for i, nivel in enumerate(tabela.columns):
        ax.barh(tabela.index, tabela[nivel], left=esquerda, color=cores[i] if cores else None, label=str(nivel))
        esquerda += tabela[nivel].to_numpy()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% das campanhas", fontsize=9, color=INK_SECUNDARIO)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=tabela.shape[1], frameon=False, fontsize=8)
    _estilo_eixo(ax)
    ax.grid(False)
    ax.set_title(titulo, fontsize=11, color=INK_PRIMARIO)
    fig.tight_layout()
    return fig


LABELS_ATRIBUTOS_MODELO = {
    "platform": "Plataforma",
    "category": "Categoria do anunciante",
    "objective": "Objetivo da campanha",
    "target_audience_age": "Faixa etária do público",
    "is_retargeting": "Retargeting",
    "video_duration_s": "Duração do vídeo",
    "format": "Formato do vídeo",
    "has_subtitle": "Legendas",
    "has_cta": "Call-to-action (CTA)",
    "has_hook": "Hook nos 3 primeiros segundos",
    "has_face": "Rosto humano no vídeo",
    "text_density": "Quantidade de texto na tela",
    "music_voice_ratio": "Proporção música/voz",
}


def feature_importance_bar(tabela_importancia: pd.DataFrame, titulo: str | None = None):
    """Gráfico de barras horizontal da permutation importance (Parte 2), no
    mesmo padrão visual da Parte 1. `tabela_importancia`: colunas `atributo`,
    `importancia_media`, `importancia_dp`, já ordenada da mais para a menos
    importante."""
    tabela = tabela_importancia.sort_values("importancia_media", ascending=True)
    rotulos = [LABELS_ATRIBUTOS_MODELO.get(a, a) for a in tabela["atributo"]]

    if titulo is None:
        top1 = LABELS_ATRIBUTOS_MODELO.get(tabela_importancia.iloc[0]["atributo"], tabela_importancia.iloc[0]["atributo"])
        titulo = f"{top1} é o atributo que mais ajuda o modelo a prever o score"

    fig, ax = plt.subplots(figsize=(8, 6))
    cores = ["#2a78d6" if v > 0 else "#c3c2b7" for v in tabela["importancia_media"]]
    ax.barh(rotulos, tabela["importancia_media"], xerr=tabela["importancia_dp"], color=cores, height=0.6, capsize=3, error_kw={"ecolor": INK_MUTED, "linewidth": 1})
    ax.axvline(0, color=INK_MUTED, linewidth=1)
    ax.set_xlabel("queda no R² ao embaralhar o atributo (importância)", fontsize=9, color=INK_SECUNDARIO)
    ax.set_title(titulo, fontsize=13, color=INK_PRIMARIO, pad=15)
    _estilo_eixo(ax)
    ax.grid(axis="x", color=GRADE, linewidth=0.8)
    _rodape(fig)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig
