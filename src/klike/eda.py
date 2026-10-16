"""Funções reutilizáveis para a EDA (Parte 1) do desafio Klike."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

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


def derived_metric_divergences(
    df: pd.DataFrame, metrica: str, numerador: str, denominador: str, tol: float = 0.01
) -> pd.DataFrame:
    """Linha a linha, retorna as divergências (erro relativo > tol) entre `metrica`
    informada e `numerador/denominador` recalculado, com colunas de contexto
    (impressions, clicks, spend, revenue, conversions) e a razão informado/recalculado.
    """
    is_zero_denom = df[denominador] == 0
    valid = df.loc[~is_zero_denom].copy()

    recalculado = valid[numerador] / valid[denominador]
    informado = valid[metrica]
    informado_zero = informado == 0

    erro_absoluto = (recalculado - informado).abs()
    erro_relativo = pd.Series(np.nan, index=valid.index)
    erro_relativo.loc[~informado_zero] = erro_absoluto.loc[~informado_zero] / informado.loc[~informado_zero].abs()

    valid["informado"] = informado
    valid["recalculado"] = recalculado
    valid["erro_relativo"] = erro_relativo
    valid["razao_informado_recalculado"] = informado / recalculado

    contexto = ["campaign_id", "impressions", "clicks", "spend", "revenue", "conversions"]
    cols = contexto + ["informado", "recalculado", "razao_informado_recalculado"]

    divergentes = valid.loc[erro_relativo > tol].assign(erro_relativo=erro_relativo.loc[erro_relativo > tol])
    return divergentes.sort_values("erro_relativo", ascending=False)[cols]


def reconstructable_cpc_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Para as linhas com `cpc` nulo, verifica se `spend` e `clicks` estão disponíveis
    (e `clicks != 0`) para reconstruir `cpc = spend/clicks`.
    """
    cpc_null = df[df["cpc"].isna()].copy()
    cpc_null["spend_disponivel"] = cpc_null["spend"].notna()
    cpc_null["clicks_disponivel"] = cpc_null["clicks"].notna()
    cpc_null["clicks_igual_zero"] = cpc_null["clicks"] == 0
    cpc_null["reconstruivel"] = (
        cpc_null["spend_disponivel"] & cpc_null["clicks_disponivel"] & ~cpc_null["clicks_igual_zero"]
    )
    cpc_null["cpc_reconstruido"] = np.where(
        cpc_null["reconstruivel"], cpc_null["spend"] / cpc_null["clicks"], np.nan
    )
    return cpc_null[
        [
            "campaign_id",
            "spend",
            "clicks",
            "spend_disponivel",
            "clicks_disponivel",
            "clicks_igual_zero",
            "reconstruivel",
            "cpc_reconstruido",
        ]
    ]


def reconstructable_revenue_nulls(df: pd.DataFrame, iqr_k: float = 1.5) -> pd.DataFrame:
    """Para as linhas com `revenue` nulo, verifica se `roas` e `spend` estão disponíveis
    para reconstruir `revenue = roas * spend`, e flagga `roas` pouco confiável
    (fora da faixa IQR da distribuição geral de `roas`, ou igual a zero).
    """
    q1, q3 = df["roas"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr

    rev_null = df[df["revenue"].isna()].copy()
    rev_null["roas_disponivel"] = rev_null["roas"].notna()
    rev_null["spend_disponivel"] = rev_null["spend"].notna()
    rev_null["roas_fora_iqr"] = (rev_null["roas"] < lower) | (rev_null["roas"] > upper)
    rev_null["roas_igual_zero"] = rev_null["roas"] == 0
    rev_null["roas_confiavel"] = ~(rev_null["roas_fora_iqr"] | rev_null["roas_igual_zero"])
    rev_null["reconstruivel"] = rev_null["roas_disponivel"] & rev_null["spend_disponivel"]
    rev_null["revenue_reconstruido"] = np.where(
        rev_null["reconstruivel"], rev_null["roas"] * rev_null["spend"], np.nan
    )
    return rev_null[
        [
            "campaign_id",
            "roas",
            "spend",
            "conversions",
            "roas_disponivel",
            "spend_disponivel",
            "roas_fora_iqr",
            "roas_igual_zero",
            "roas_confiavel",
            "reconstruivel",
            "revenue_reconstruido",
        ]
    ]


def revenue_zero_breakdown(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Contagem e % de `revenue == 0` (entre as linhas com `revenue` não nulo) por `group_col`."""
    comparavel = df[df["revenue"].notna()]
    is_zero = comparavel["revenue"] == 0

    tabela = pd.crosstab(comparavel[group_col], is_zero)
    tabela.columns = ["revenue>0", "revenue==0"]
    tabela["total"] = tabela.sum(axis=1)
    tabela["pct_zero"] = (100 * tabela["revenue==0"] / tabela["total"]).round(1)
    return tabela


def null_cooccurrence_matrix(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    """Matriz de co-ocorrência de nulos entre `cols`: contagem observada, esperada
    sob independência (nA*nB/n) e a razão observado/esperado.
    """
    is_null = df[cols].isna()
    n = len(df)

    observado = pd.DataFrame(index=cols, columns=cols, dtype=float)
    esperado = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c1 in cols:
        for c2 in cols:
            observado.loc[c1, c2] = (is_null[c1] & is_null[c2]).sum()
            esperado.loc[c1, c2] = is_null[c1].sum() * is_null[c2].sum() / n

    razao = observado / esperado

    return {
        "observado": observado.astype(int),
        "esperado_sob_independencia": esperado.round(2),
        "razao_observado_esperado": razao.round(2),
        "nulos_por_linha": is_null.sum(axis=1).value_counts().sort_index().rename("n_linhas").to_frame(),
    }


def group_size_table(df: pd.DataFrame, col1: str, col2: str, min_n: int = 20):
    """Crosstab col1 x col2 com destaque (via Styler) das células com n < min_n."""
    tabela = pd.crosstab(df[col1], df[col2])

    def _highlight(valor: int) -> str:
        return "background-color: #ffcccc" if valor < min_n else ""

    return tabela.style.map(_highlight)


# ---------------------------------------------------------------------------
# Parte 1 — Insights (perguntas 2-4): atributos acionáveis x performance,
# padrões por plataforma.
# ---------------------------------------------------------------------------


def discretize_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona `video_duration_bin` (`<=15s`, `15-30s`, `>30s`; NaN se
    `video_duration_s` for nulo).

    Faixas ajustadas de 4 para 3: a faixa `>60s` isolada tinha n=8 (<20,
    amostra insuficiente); unida a `30-60s` em `>30s` (n=56). Ver
    `docs/eda_insights.md`.
    """
    df = df.copy()
    bins = [0, 15, 30, np.inf]
    labels = ["<=15s", "15-30s", ">30s"]
    df["video_duration_bin"] = pd.cut(df["video_duration_s"], bins=bins, labels=labels, right=True)
    return df


def discretize_music_voice(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona `music_voice_bin` (`mais_voz` 0–0.33, `equilibrado` 0.33–0.66,
    `mais_musica` 0.66–1; NaN se `music_voice_ratio` for nulo)."""
    df = df.copy()
    bins = [-0.001, 0.33, 0.66, 1.0]
    labels = ["mais_voz", "equilibrado", "mais_musica"]
    df["music_voice_bin"] = pd.cut(df["music_voice_ratio"], bins=bins, labels=labels, right=True)
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona `watch_rate` (`avg_watch_time_s / video_duration_s`, fração do
    vídeo assistida — mais justa que `avg_watch_time_s` absoluto para comparar
    vídeos de durações diferentes) e `cvr` (`conversions / clicks`)."""
    df = df.copy()
    df["watch_rate"] = df["avg_watch_time_s"] / df["video_duration_s"]
    df["cvr"] = df["conversions"] / df["clicks"]
    return df


def roas_subset_mask(df: pd.DataFrame) -> pd.Series:
    """Máscara para análises de `roas`/`revenue`: exclui `category`=`Branding`
    (zero estrutural, ver `docs/eda_qualidade.md` seção 4) e restringe a
    `objective` em {`conversions`, `traffic`, `app_install`} (objetivos com
    receita atribuída esperada)."""
    return (df["category"] != "Branding") & df["objective"].isin(["conversions", "traffic", "app_install"])


def bootstrap_median_lift(
    valores_referencia,
    valores_grupo,
    tipo_efeito: str = "relativo",
    n_boot: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compara, via bootstrap, a mediana de `valores_grupo` (grupo comparado)
    com a de `valores_referencia` (grupo de referência).

    **Reamostragem:** cada grupo é reamostrado SEPARADAMENTE, com reposição,
    preservando o próprio `n` de cada grupo em cada uma das `n_boot`
    reamostragens — bootstrap de duas amostras independentes, não pareado.

    `tipo_efeito`:
    - `"relativo"`: efeito = lift % = `(mediana_grupo/mediana_referencia - 1) * 100`.
      Se a mediana OBSERVADA do grupo de referência for 0, o lift relativo não
      é definido: a função retorna `efeito_relatado=NaN` em `lift_pct`,
      `motivo="mediana de referência = 0"`, e passa a reportar a DIFERENÇA
      ABSOLUTA (sempre calculada, em `diferenca_absoluta` e seu IC) como o
      efeito primário (`efeito_relatado`/`ic_baixo`/`ic_alto`). Se mais de 5%
      das reamostragens tiverem mediana de referência = 0 (mesmo quando a
      observada não é 0), `ic_confiavel=False`.
    - `"absoluto"`: efeito = diferença de medianas em pontos
      (`mediana_grupo - mediana_referencia`). Usado para `klike_score`.

    Retorna um dict com `n_referencia`, `n_grupo`, `mediana_referencia`,
    `mediana_grupo`, `tipo_efeito_usado`, `efeito_relatado`, `ic_baixo`,
    `ic_alto`, `evidencia` (IC não cruza 0), `ic_confiavel`,
    `frac_ref_zero_bootstrap`, `motivo`, `diferenca_absoluta`,
    `diferenca_absoluta_ic_baixo`, `diferenca_absoluta_ic_alto`.
    """
    rng = np.random.default_rng(random_state)
    ref = np.asarray(valores_referencia, dtype=float)
    grp = np.asarray(valores_grupo, dtype=float)
    n_ref, n_grp = len(ref), len(grp)

    mediana_ref = float(np.median(ref))
    mediana_grp = float(np.median(grp))
    diferenca_absoluta = mediana_grp - mediana_ref

    idx_ref = rng.integers(0, n_ref, size=(n_boot, n_ref))
    idx_grp = rng.integers(0, n_grp, size=(n_boot, n_grp))
    med_ref_boot = np.median(ref[idx_ref], axis=1)
    med_grp_boot = np.median(grp[idx_grp], axis=1)

    diff_boot = med_grp_boot - med_ref_boot
    diff_ic_baixo, diff_ic_alto = np.percentile(diff_boot, [2.5, 97.5])
    frac_ref_zero_boot = float(np.mean(med_ref_boot == 0))

    resultado = {
        "n_referencia": n_ref,
        "n_grupo": n_grp,
        "mediana_referencia": mediana_ref,
        "mediana_grupo": mediana_grp,
        "diferenca_absoluta": diferenca_absoluta,
        "diferenca_absoluta_ic_baixo": float(diff_ic_baixo),
        "diferenca_absoluta_ic_alto": float(diff_ic_alto),
        "frac_ref_zero_bootstrap": frac_ref_zero_boot,
        "motivo": None,
    }

    if tipo_efeito == "absoluto":
        resultado["tipo_efeito_usado"] = "absoluto"
        resultado["efeito_relatado"] = diferenca_absoluta
        resultado["ic_baixo"] = float(diff_ic_baixo)
        resultado["ic_alto"] = float(diff_ic_alto)
        resultado["ic_confiavel"] = True
    elif mediana_ref == 0:
        resultado["tipo_efeito_usado"] = "absoluto (fallback: ref=0)"
        resultado["efeito_relatado"] = diferenca_absoluta
        resultado["ic_baixo"] = float(diff_ic_baixo)
        resultado["ic_alto"] = float(diff_ic_alto)
        resultado["motivo"] = "mediana de referência = 0"
        resultado["ic_confiavel"] = True
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            lift_boot = np.where(med_ref_boot != 0, (med_grp_boot / med_ref_boot - 1) * 100, np.nan)
        lift_boot_validos = lift_boot[~np.isnan(lift_boot)]
        lift_ic_baixo, lift_ic_alto = np.percentile(lift_boot_validos, [2.5, 97.5])
        lift_pct = (mediana_grp / mediana_ref - 1) * 100

        resultado["tipo_efeito_usado"] = "relativo"
        resultado["efeito_relatado"] = lift_pct
        resultado["ic_baixo"] = float(lift_ic_baixo)
        resultado["ic_alto"] = float(lift_ic_alto)
        resultado["ic_confiavel"] = frac_ref_zero_boot <= 0.05

    resultado["evidencia"] = bool(resultado["ic_baixo"] > 0 or resultado["ic_alto"] < 0)
    return resultado


def effect_by_attribute(
    df: pd.DataFrame,
    attr_col: str,
    metric_col: str,
    nivel_referencia,
    ordem_niveis: list | None = None,
    n_boot: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Para cada nível de `attr_col` (exceto a referência), compara a mediana
    de `metric_col` com a do nível de referência `nivel_referencia`, via
    `bootstrap_median_lift`.

    Exclusão caso a caso: linhas com `attr_col` ou `metric_col` nulos são
    descartadas só para esta comparação (análise de casos disponíveis). Usa
    efeito relativo (%) para todas as métricas, exceto `klike_score`, que usa
    a diferença absoluta de medianas em pontos (regra do CLAUDE.md/decisão do
    usuário: score não é uma razão, variação em pontos é mais interpretável).
    """
    tipo_efeito = "absoluto" if metric_col == "klike_score" else "relativo"

    dados = df[[attr_col, metric_col]].dropna()
    valores_ref = dados.loc[dados[attr_col] == nivel_referencia, metric_col].to_numpy()

    niveis_presentes = [v for v in dados[attr_col].unique() if v != nivel_referencia]
    if ordem_niveis is not None:
        niveis = [v for v in ordem_niveis if v in niveis_presentes]
    else:
        niveis = sorted(niveis_presentes, key=str)

    linhas = [
        {
            "atributo": attr_col,
            "nivel": nivel_referencia,
            "e_referencia": True,
            "n": len(valores_ref),
            "mediana": float(np.median(valores_ref)) if len(valores_ref) else np.nan,
            "tipo_efeito": tipo_efeito,
            "efeito": np.nan,
            "ic_baixo": np.nan,
            "ic_alto": np.nan,
            "evidencia": False,
            "ic_confiavel": True,
            "motivo": None,
        }
    ]

    for nivel in niveis:
        valores_grupo = dados.loc[dados[attr_col] == nivel, metric_col].to_numpy()
        if len(valores_grupo) == 0:
            continue
        res = bootstrap_median_lift(
            valores_ref, valores_grupo, tipo_efeito=tipo_efeito, n_boot=n_boot, random_state=random_state
        )
        linhas.append(
            {
                "atributo": attr_col,
                "nivel": nivel,
                "e_referencia": False,
                "n": res["n_grupo"],
                "mediana": res["mediana_grupo"],
                "tipo_efeito": res["tipo_efeito_usado"],
                "efeito": res["efeito_relatado"],
                "ic_baixo": res["ic_baixo"],
                "ic_alto": res["ic_alto"],
                "evidencia": res["evidencia"],
                "ic_confiavel": res["ic_confiavel"],
                "motivo": res["motivo"],
            }
        )

    tabela = pd.DataFrame(linhas)
    tabela["metrica"] = metric_col
    tabela["n_referencia"] = len(valores_ref)
    return tabela


def spearman_table(df: pd.DataFrame, pares: list[tuple[str, str]]) -> pd.DataFrame:
    """Correlação de Spearman (rho, p-valor, n) para cada par `(x, y)` em
    `pares`, com exclusão caso a caso de nulos."""
    linhas = []
    for x_col, y_col in pares:
        dados = df[[x_col, y_col]].dropna()
        rho, p = spearmanr(dados[x_col], dados[y_col])
        linhas.append({"x": x_col, "y": y_col, "n": len(dados), "rho": round(float(rho), 4), "p_valor": round(float(p), 4)})
    return pd.DataFrame(linhas)


def classificar_robustez(*tabelas_efeito: pd.DataFrame) -> pd.DataFrame:
    """Recebe uma ou mais tabelas produzidas por `effect_by_attribute`
    (concatenadas; podem incluir uma coluna opcional `platform` para as
    tabelas por plataforma) e classifica cada achado com evidência:

    - **robusto**: IC não cruza 0 E o mesmo (`atributo`, `nivel`) tem evidência
      com o MESMO SINAL em pelo menos mais uma métrica (olhando as linhas sem
      `platform`, i.e. a análise geral) OU em pelo menos 2 plataformas
      distintas para a mesma métrica (olhando as linhas com `platform`).
    - **isolado**: tem evidência, mas em uma única combinação
      atributo×nível×métrica(×plataforma), sem repetição.

    Retorna só as linhas com `evidencia=True`, com uma coluna `classificacao`
    adicional, ordenadas com os "robusto" primeiro.
    """
    todas = pd.concat(tabelas_efeito, ignore_index=True)
    if "platform" not in todas.columns:
        todas["platform"] = pd.NA

    todas["sinal"] = np.sign(todas["efeito"].astype(float))
    evidencia = todas.loc[todas["evidencia"]].copy()

    geral = todas.loc[todas["platform"].isna()]
    por_plataforma = todas.loc[todas["platform"].notna()]

    def _classificar(row) -> str:
        outras_metricas = geral.loc[
            (geral["atributo"] == row["atributo"])
            & (geral["nivel"] == row["nivel"])
            & (geral["metrica"] != row["metrica"])
            & geral["evidencia"]
            & (geral["sinal"] == row["sinal"])
        ]
        if len(outras_metricas) >= 1:
            return "robusto"

        mesma_metrica_outras_plataformas = por_plataforma.loc[
            (por_plataforma["atributo"] == row["atributo"])
            & (por_plataforma["nivel"] == row["nivel"])
            & (por_plataforma["metrica"] == row["metrica"])
            & por_plataforma["evidencia"]
            & (por_plataforma["sinal"] == row["sinal"])
        ]
        if mesma_metrica_outras_plataformas["platform"].nunique() >= 2:
            return "robusto"

        return "isolado"

    evidencia["classificacao"] = evidencia.apply(_classificar, axis=1)
    ordem = {"robusto": 0, "isolado": 1}
    evidencia["_ordem"] = evidencia["classificacao"].map(ordem)
    evidencia = evidencia.sort_values(["_ordem", "atributo", "nivel"]).drop(columns="_ordem")
    return evidencia


def bootstrap_median_ci(valores, n_boot: int = 1000, random_state: int = 42) -> dict:
    """IC95% da mediana de UM grupo só (sem comparação com referência), por
    bootstrap com reposição. Usado quando o que importa é a incerteza da
    própria mediana por faixa (ex.: CTR mediano por faixa de `n_boas_praticas`),
    não um lift entre dois grupos — ver `bootstrap_median_lift` para isso.

    Retorna `n`, `mediana`, `ic_baixo`, `ic_alto`.
    """
    rng = np.random.default_rng(random_state)
    valores = np.asarray(valores, dtype=float)
    n = len(valores)

    mediana = float(np.median(valores))
    idx = rng.integers(0, n, size=(n_boot, n))
    medianas_boot = np.median(valores[idx], axis=1)
    ic_baixo, ic_alto = np.percentile(medianas_boot, [2.5, 97.5])

    return {"n": n, "mediana": mediana, "ic_baixo": float(ic_baixo), "ic_alto": float(ic_alto)}
