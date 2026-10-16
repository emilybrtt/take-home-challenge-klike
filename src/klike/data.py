"""Carregamento e limpeza dos dados brutos do desafio Klike."""

from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "klike_challenge_dataset.csv"
CLEAN_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "klike_clean.parquet"

BOOL_COLS = ["is_retargeting", "has_subtitle", "has_cta", "has_hook", "has_face"]
CATEGORY_COLS = ["platform", "category", "objective", "target_audience_age", "format", "text_density"]

# campaign_ids com dado bruto corrompido, identificados na investigação de
# qualidade (ver docs/eda_qualidade.md, seções 3 e 4).
IMPRESSIONS_CORROMPIDAS = ["KLK-0148", "KLK-0232", "KLK-0258", "KLK-0413"]
SPEND_CORROMPIDO = ["KLK-0259", "KLK-0109", "KLK-0224"]
CPC_INFORMADO_CORROMPIDO = ["KLK-0100", "KLK-0416", "KLK-0240"]
REVENUE_AMBIGUO = ["KLK-0059", "KLK-0110"]


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Lê o CSV bruto do desafio com os tipos de coluna corretos.

    - `date` como datetime
    - colunas booleanas como "boolean" (nullable): `astype(bool)` converteria
      `NaN` silenciosamente para `True` em colunas com nulos (ex.: `has_subtitle`)
    - colunas categóricas como category
    """
    df = pd.read_csv(path, parse_dates=["date"])

    for col in BOOL_COLS:
        df[col] = df[col].astype("boolean")

    for col in CATEGORY_COLS:
        df[col] = df[col].astype("category")

    return df


def _marcar(df: pd.DataFrame, mask: pd.Series, nota: str) -> None:
    """Marca `linha_corrigida=True` e acrescenta `nota` a `correcao_aplicada`,
    sem sobrescrever notas já registradas (uma linha pode acumular mais de uma correção).
    """
    df.loc[mask, "linha_corrigida"] = True
    tem_nota_anterior = mask & (df["correcao_aplicada"] != "")
    df.loc[tem_nota_anterior, "correcao_aplicada"] += "; " + nota
    df.loc[mask & ~tem_nota_anterior, "correcao_aplicada"] = nota


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica correções determinísticas, linha a linha, aos erros de dado bruto
    identificados na investigação de qualidade (`docs/eda_qualidade.md`, seções
    3 e 4). Função pura: recebe o df de `load_raw()` e retorna uma cópia
    corrigida — não altera o df de entrada. A imputação estatística de nulos
    (colunas que não têm correção determinística possível) fica para o
    `Pipeline` de modelagem (Parte 2), não é feita aqui.

    Regra geral: a fonte de verdade é o valor **consistente com mais relações
    entre colunas** (triangulação: ctr/clicks/impressions, cpc/spend/clicks,
    roas/revenue/spend), não automaticamente a coluna bruta.

    Correções aplicadas, em ordem:

    1. **`impressions` corrompido** (`KLK-0148`, `KLK-0232`, `KLK-0258`,
       `KLK-0413`): esses 4 valores de `impressions` estão 1–2 ordens de
       magnitude acima de qualquer outra linha (~12–14 milhões vs. máximo de
       1.57 milhão no resto do dataset), enquanto `clicks` e `ctr` informados
       são plausíveis e consistentes com o restante dos dados. Reconstruído
       como `impressions = round(clicks / ctr_informado)`.

    2. **`spend` corrompido** (`KLK-0259`, `KLK-0109`, `KLK-0224`): `cpc` e
       `roas` informados concordam entre si (e com `revenue` informado, para
       as duas primeiras), mas discordam de `spend`. Reconstruído como
       `spend = cpc_informado * clicks`. Para `KLK-0224` (`revenue`=0), não há
       como confirmar via `roas` (que é 0 independente de `spend`) — limitação
       registrada em `correcao_aplicada`.

    3. **`cpc` informado corrompido** (`KLK-0100`, `KLK-0416`, `KLK-0240`):
       `spend` e `clicks` são plausíveis; só o `cpc` informado estava errado.
       Corrigido pelo recálculo do passo 5 (não há correção de coluna bruta
       aqui, só o registro da marcação).

    4. **`revenue` nulo** (25 linhas): reconstruído como
       `revenue = roas_informado * spend` (usando o `spend` já corrigido pelo
       passo 2, quando aplicável). Outlier de `roas` não é tratado como dado
       ruim (não é motivo para deixar de reconstruir). Duas linhas
       (`KLK-0059`, `KLK-0110`) têm `roas`=0 com `conversions`>0 — ambíguo
       (receita zero com conversões registradas) — marcadas com nota extra,
       mas reconstruídas com o mesmo critério (`revenue`=0).

    5. **`ctr`, `cpc` e `roas` recalculados em TODAS as linhas**, a partir das
       colunas brutas já corrigidas (`clicks/impressions`, `spend/clicks`,
       `revenue/spend`). Isso elimina o ruído de arredondamento do valor
       informado e garante consistência interna — inclusive preenche o `cpc`
       das 28 linhas onde ele era nulo (`spend` e `clicks` sempre disponíveis
       nessas linhas). Não é marcado como correção para a maioria das linhas
       (a mudança é desprezível, só arredondamento); é marcado para as 3
       linhas do passo 3 (valor informado estava de fato errado) e para as 28
       linhas onde `cpc` era nulo.

    Nulos em `has_subtitle`, `video_duration_s`, `music_voice_ratio`,
    `avg_watch_time_s` e `engagement_rate` **não são tratados aqui** — ficam
    para o Pipeline de modelagem (imputação + indicador de nulo) ou para
    análise de casos disponíveis.

    Adiciona duas colunas: `linha_corrigida` (bool) e `correcao_aplicada`
    (texto; vazio quando nenhuma correção foi aplicada à linha).
    """
    df = df.copy()
    df["linha_corrigida"] = False
    df["correcao_aplicada"] = ""

    # 1. impressions corrompido: reconstruído a partir de clicks e ctr informados
    mask_impressions = df["campaign_id"].isin(IMPRESSIONS_CORROMPIDAS)
    df.loc[mask_impressions, "impressions"] = (
        (df.loc[mask_impressions, "clicks"] / df.loc[mask_impressions, "ctr"]).round().astype(int)
    )
    _marcar(df, mask_impressions, "impressions reconstruído = round(clicks / ctr_informado)")

    # 2. spend corrompido: reconstruído a partir de cpc_informado e clicks
    mask_spend = df["campaign_id"].isin(SPEND_CORROMPIDO)
    df.loc[mask_spend, "spend"] = df.loc[mask_spend, "cpc"] * df.loc[mask_spend, "clicks"]
    _marcar(df, mask_spend, "spend reconstruído = cpc_informado * clicks")
    mask_sem_confirmacao = df["campaign_id"] == "KLK-0224"
    _marcar(df, mask_sem_confirmacao, "limitação: revenue=0, não foi possível confirmar via roas")

    # 3. cpc informado corrompido: sem correção de coluna bruta, só a marcação
    #    (o recálculo do passo 5 substitui o valor errado)
    mask_cpc_informado = df["campaign_id"].isin(CPC_INFORMADO_CORROMPIDO)
    _marcar(df, mask_cpc_informado, "cpc informado estava corrompido; substituído por spend/clicks")

    # 4. revenue nulo: reconstruído a partir de roas_informado e spend (já corrigido)
    mask_revenue_nulo = df["revenue"].isna()
    df.loc[mask_revenue_nulo, "revenue"] = df.loc[mask_revenue_nulo, "roas"] * df.loc[mask_revenue_nulo, "spend"]
    _marcar(df, mask_revenue_nulo, "revenue reconstruído = roas_informado * spend")
    mask_revenue_ambiguo = mask_revenue_nulo & df["campaign_id"].isin(REVENUE_AMBIGUO)
    _marcar(df, mask_revenue_ambiguo, "ambíguo: roas informado=0 com conversions>0")

    # 5. recálculo global de ctr, cpc e roas a partir das colunas brutas corrigidas
    mask_cpc_nulo = df["cpc"].isna()
    _marcar(df, mask_cpc_nulo, "cpc reconstruído = spend / clicks (estava nulo)")

    df["ctr"] = df["clicks"] / df["impressions"]
    df["cpc"] = df["spend"] / df["clicks"]
    df["roas"] = df["revenue"] / df["spend"]

    return df


def load_clean(raw_path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """`load_raw()` + `clean()`: carrega o dataset já com as correções de
    qualidade aplicadas."""
    return clean(load_raw(raw_path))
