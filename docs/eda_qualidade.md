# Qualidade dos dados — EDA (Parte 1, pergunta 1)

Diagnóstico de qualidade do dataset `data/raw/klike_challenge_dataset.csv` (500 linhas). Nenhum dado foi alterado nesta etapa — apenas diagnóstico. Análise completa, com código, em `notebooks/01_eda.ipynb`.

## 1. Resumo dos achados

### 1.1 Valores faltantes e bug corrigido

Nulos por coluna (CSV cru, idêntico ao resultado de `load_raw()` após a correção):

| Coluna | Nulos |
| :--- | ---: |
| `has_subtitle` | 46 |
| `video_duration_s` | 39 |
| `music_voice_ratio` | 38 |
| `engagement_rate` | 32 |
| `cpc` | 28 |
| `avg_watch_time_s` | 27 |
| `revenue` | 25 |
| demais colunas | 0 |

**Bug encontrado e corrigido:** `load_raw()` convertia as colunas booleanas com `.astype(bool)`, que transforma `NaN` silenciosamente em `True`. Isso afetava `has_subtitle` (única coluna booleana com nulos). Corrigido para `.astype("boolean")` (dtype nullable do pandas), que preserva os 46 nulos como `<NA>`.

**Os nulos de `has_subtitle` são aleatórios?** Comparação entre o grupo nulo (n=46) e não nulo (n=454):

| | LinkedIn | Meta | TikTok |
| :--- | ---: | ---: | ---: |
| nulo (%) | 19.6 | 45.7 | 34.8 |
| não nulo (%) | 16.3 | 48.5 | 35.2 |

| | App Install | Branding | E-commerce | Lead Gen | SaaS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| nulo (%) | 21.7 | 17.4 | 43.5 | 8.7 | 8.7 |
| não nulo (%) | 15.9 | 12.1 | 31.9 | 19.6 | 20.5 |

| | horizontal | quadrado | vertical |
| :--- | ---: | ---: | ---: |
| nulo (%) | 21.7 | 26.1 | 52.2 |
| não nulo (%) | 22.2 | 23.8 | 54.0 |

| | klike_score (mediana) | ctr (mediana) | engagement_rate (mediana) |
| :--- | ---: | ---: | ---: |
| nulo (n=46) | 61.50 | 0.047 | 0.026 |
| não nulo (n=454) | 60.35 | 0.065 | 0.021 |

`platform` e `format` têm distribuição quase idêntica entre os dois grupos (diferenças ≤ 3 p.p.). `category` mostra uma diferença maior — o grupo nulo tem mais `E-commerce` (+11.6 p.p.) e menos `SaaS`/`Lead Gen` (-11.8 p.p. e -10.9 p.p.) — mas com n=46 essa diferença é pouco confiável. `klike_score` é praticamente igual entre os grupos; `ctr` é 27.7% menor (relativo) no grupo nulo. **Conclusão:** nulos parecem majoritariamente aleatórios (MAR/MCAR), com sinal fraco e pouco confiável de associação com `category`/`ctr`.

### 1.2 Duplicatas

- Linhas totalmente duplicadas: **0**
- `campaign_id` duplicado: **0**

### 1.3 Valores categóricos

Todos os valores estão dentro do dicionário de dados (`docs/CHALLENGE.md`) — nenhuma grafia inconsistente encontrada.

| Coluna | Valores (contagem) |
| :--- | :--- |
| `platform` | Meta=241, TikTok=176, LinkedIn=83 |
| `category` | E-commerce=165, SaaS=97, Lead Gen=93, App Install=82, Branding=63 |
| `objective` | conversions=165, traffic=150, engagement=79, awareness=60, app_install=46 |
| `target_audience_age` | 25-34=143, 35-44=139, 18-24=137, 45+=81 |
| `format` | vertical=269, quadrado=120, horizontal=111 |
| `text_density` | medium=230, low=179, high=91 |

### 1.4 Faixas válidas

Nenhuma violação em nenhuma das 11 regras checadas:

| Regra | Linhas violadas | % |
| :--- | ---: | ---: |
| `klike_score` fora de [0, 100] | 0 | 0.0 |
| `music_voice_ratio` fora de [0, 1] | 0 | 0.0 |
| `ctr` fora de [0, 1] | 0 | 0.0 |
| `engagement_rate` fora de [0, 1] | 0 | 0.0 |
| `impressions` negativo | 0 | 0.0 |
| `clicks` negativo | 0 | 0.0 |
| `conversions` negativo | 0 | 0.0 |
| `spend` negativo | 0 | 0.0 |
| `revenue` negativo | 0 | 0.0 |
| `clicks` > `impressions` | 0 | 0.0 |
| `conversions` > `clicks` | 0 | 0.0 |

### 1.5 Consistência das métricas derivadas

Recalculamos `ctr = clicks/impressions`, `cpc = spend/clicks` e `roas = revenue/spend`, comparando com o valor informado via **erro relativo** (`|recalculado - informado| / |informado|`, limiar de 1%).

| Métrica | Excluídas (denominador = 0) | Excluídas (informado = 0) | n comparável | Divergentes (> 1% relativo) | % divergente | Erro absoluto médio | Erro absoluto máximo |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ctr` | 0 | 0 | 500 | 16 | 3.20% | 0.000293 | 0.042312 |
| `cpc` | 0 | 0 | 500 | 6 | 1.20% | 10.820754 | 4790.314500 |
| `roas` | 0 | 76 | 424 | 27 | 6.37% | 0.031006 | 13.449242 |

- Denominador zero (não ocorre neste dataset): `impressions`=0 → 0 linhas; `clicks`=0 → 0 linhas; `spend`=0 → 0 linhas.
- `revenue` informado = 0 em 76 linhas: torna o erro relativo do `roas` indefinido (divisão por zero no denominador do erro), por isso essas linhas foram excluídas apenas do cálculo de erro relativo do `roas` (não da base de dados).
- `avg_watch_time_s > video_duration_s`: **0 linhas (0.0%)**.

### 1.6 Datas

- Intervalo: **2024-01-01** a **2024-12-30**.
- Volume de campanhas por mês relativamente estável entre as três plataformas (ver `notebooks/01_eda.ipynb`, seção 6, e figura `reports/figures/campanhas_por_mes_e_plataforma.png`).

### 1.7 Outliers e forma da distribuição

Critério de outlier: IQR (1.5×IQR). Critério de log-normalidade: skewness original > 1 **e** `|skewness(log1p(x))| < 0.5 * |skewness original|`.

| Coluna | Skewness | Skewness (log1p) | % outliers (IQR) | Parece log-normal? |
| :--- | ---: | ---: | ---: | :---: |
| `video_duration_s` | 2.547 | 0.262 | 3.69% | sim |
| `music_voice_ratio` | 0.245 | -0.021 | 0.00% | não |
| `impressions` | 10.966 | 0.727 | 9.00% | sim |
| `clicks` | 5.153 | -0.210 | 9.80% | sim |
| `ctr` | 0.434 | 0.385 | 0.00% | não |
| `cpc` | 9.894 | 2.745 | 13.98% | sim |
| `spend` | 5.926 | 0.042 | 12.20% | sim |
| `conversions` | 6.016 | -0.004 | 12.20% | sim |
| `revenue` | 8.317 | -0.975 | 13.68% | sim |
| `roas` | 4.390 | 0.576 | 5.40% | sim |
| `avg_watch_time_s` | 4.002 | 0.189 | 4.44% | sim |
| `engagement_rate` | 1.303 | 1.249 | 4.27% | não (queda de skew < 50%) |
| `klike_score` | -0.225 | -1.336 | 0.40% | não |

Variáveis log-normais (9): `video_duration_s`, `impressions`, `clicks`, `cpc`, `spend`, `conversions`, `revenue`, `roas`, `avg_watch_time_s`.

### 1.8 Tamanho dos grupos

**`platform × category`** (destaque = n < 20):

| platform | App Install | Branding | E-commerce | Lead Gen | SaaS |
| :--- | ---: | ---: | ---: | ---: | ---: |
| LinkedIn | **17** | **9** | 28 | **13** | **16** |
| Meta | 38 | 30 | 79 | 45 | 49 |
| TikTok | 27 | 24 | 58 | 35 | 32 |

**`platform × objective`** (destaque = n < 20):

| platform | app_install | awareness | conversions | engagement | traffic |
| :--- | ---: | ---: | ---: | ---: | ---: |
| LinkedIn | **7** | **10** | 26 | **10** | 30 |
| Meta | 21 | 24 | 84 | 45 | 67 |
| TikTok | **18** | 26 | 55 | 24 | 53 |

Combinações com **n < 20** (8 no total): LinkedIn×App Install (17), LinkedIn×Branding (9), LinkedIn×Lead Gen (13), LinkedIn×SaaS (16), LinkedIn×app_install (7), LinkedIn×awareness (10), LinkedIn×engagement (10), TikTok×app_install (18).

---

## 2. Tabela de tratamentos propostos

Nenhum tratamento foi aplicado nesta etapa — apenas diagnóstico. Os tratamentos abaixo são propostas para a etapa de modelagem (Parte 2), a serem implementadas dentro de um `Pipeline`/`ColumnTransformer` do sklearn (regra c do `CLAUDE.md`).

| Problema | Linhas afetadas | Tratamento proposto | Justificativa |
| :--- | :--- | :--- | :--- |
| Bug: nulos de `has_subtitle` virando `True` via `.astype(bool)` | 46 | Corrigido em `load_raw()` para `.astype("boolean")`; na modelagem, imputar como categoria "desconhecido" ou pela moda, dentro do Pipeline | Nulos parecem majoritariamente aleatórios (MAR/MCAR); um valor booleano inferido errado silenciosamente é pior que um nulo explícito |
| Nulos em `video_duration_s` (39), `music_voice_ratio` (38), `engagement_rate` (32), `cpc` (28), `avg_watch_time_s` (27), `revenue` (25) | 39, 38, 32, 28, 27, 25 | `SimpleImputer` (mediana) dentro de `ColumnTransformer`, nunca aplicado manualmente antes do split/CV | Regra (c) do CLAUDE.md — evitar vazamento de dados; mediana é robusta às distribuições assimétricas identificadas na seção 1.7 |
| Divergência entre métrica informada e recalculada: `ctr` 16/500 (3.20%), `cpc` 6/500 (1.20%), `roas` 27/424 (6.37%) | 16 / 6 / 27 | Manter o valor informado como fonte de verdade (é o que provavelmente alimentou o `klike_score`); não remover linhas, apenas documentar a divergência | Em dataset de 500 linhas, descartar dados por pequenas divergências (prováveis arredondamentos) tem custo alto e benefício incerto |
| `revenue` informado = 0 torna o erro relativo do `roas` indefinido | 76 | Excluir essas linhas apenas do cálculo do erro relativo (já feito); manter os dados originais para modelagem | Divisão por zero aqui é uma condição matemática esperada (sem receita), não um erro de dado |
| Ausência de denominador zero em `impressions`, `clicks`, `spend` | 0 (nenhuma linha neste dataset) | Nenhum tratamento necessário agora; manter guard de divisão por zero no código (feature engineering / recommender) | Não ocorre nos dados atuais, mas pode ocorrer em dados futuros reais |
| Variáveis fortemente assimétricas / log-normais: `impressions` (skew 10.97), `clicks` (5.15), `cpc` (9.89), `spend` (5.93), `conversions` (6.02), `revenue` (8.32), `roas` (4.39), `avg_watch_time_s` (4.00), `video_duration_s` (2.55) | ver seção 1.7 | Aplicar `log1p` na etapa de feature engineering para modelos lineares; modelos baseados em árvore podem usar as variáveis originais | `log1p` reduz a assimetria em mais de 50% nessas colunas, diminuindo a influência de outliers em modelos sensíveis a escala |
| Outliers pelo critério IQR: até 13.98% das linhas (`cpc`), 13.68% (`revenue`), 12.20% (`spend`, `conversions`) | ver tabela da seção 1.7 (0.00%–13.98%) | Não remover linhas; usar `RobustScaler` dentro do Pipeline e/ou preferir modelos robustos a outliers (árvores) | Remover 12–14% das 500 linhas é um custo alto de informação; os valores parecem extremos legítimos (campanhas de maior gasto/alcance), não erros de digitação |
| `engagement_rate` assimétrico (skew 1.303) mas não "log-normal limpo" (skew log1p 1.249, queda < 50%) | n=500 | Avaliar `sqrt`/Box-Cox como alternativa ao `log1p`, ou manter como está e usar modelo robusto a assimetria | `log1p` não atende ao critério de log-normalidade definido |
| Subgrupos com n < 20 em `platform × category`/`platform × objective` | LinkedIn×App Install=17, LinkedIn×Branding=9, LinkedIn×Lead Gen=13, LinkedIn×SaaS=16, LinkedIn×app_install=7, LinkedIn×awareness=10, LinkedIn×engagement=10, TikTok×app_install=18 | Sempre reportar o `n` do subgrupo ao gerar recomendações contextuais (regra e do CLAUDE.md); considerar agregar categorias quando `n` for muito pequeno | Estimativas de efeito com `n<20` têm alta variância e podem levar o Recommendations Engine (Parte 3) a sugerir mudanças erradas |

---

## 3. Investigação complementar

Diagnóstico adicional sobre as divergências e nulos das seções 1.5 e 1.1/1.7, para decidir se os erros são arredondamento (inofensivo) ou dado corrompido/trocado (precisa de tratamento específico). **Nenhum dado foi alterado.** Código completo em `notebooks/01_eda.ipynb`, seção 9.

### 3.1 Linhas divergentes (ctr, cpc, roas)

**`ctr` — 16 linhas divergentes**, ordenadas por razão informado/recalculado:

| campaign_id | impressions | clicks | spend | revenue | conversions | ctr informado | ctr recalculado | razão inf./recalc. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KLK-0232 | 12 594 257 | 345 | 162.15 | 77.59 | 2 | 0.0295 | 0.000027 | **1076.90** |
| KLK-0148 | 13 007 321 | 629 | 339.66 | 0.00 | 2 | 0.0148 | 0.000048 | **306.05** |
| KLK-0413 | 14 052 761 | 2695 | 9243.85 | 2227.32 | 259 | 0.0374 | 0.000192 | **195.02** |
| KLK-0258 | 12 873 562 | 3704 | 2037.20 | 668.83 | 107 | 0.0426 | 0.000288 | **148.06** |
| KLK-0254 | 1962 | 4 | 2.96 | 0.00 | 0 | 0.0022 | 0.002039 | 1.08 |
| KLK-0241 | 3385 | 38 | 55.86 | 0.00 | 0 | 0.0115 | 0.011226 | 1.02 |
| KLK-0068 | 3679 | 27 | 13.50 | 0.00 | 0 | 0.0075 | 0.007339 | 1.02 |
| KLK-0041 | 1000 | 23 | 27.14 | 50.98 | 1 | 0.0234 | 0.023000 | 1.02 |
| KLK-0466 | 9222 | 59 | 53.69 | NaN | 0 | 0.0065 | 0.006398 | 1.02 |
| KLK-0415 | 1038 | 32 | 21.76 | 14.04 | 2 | 0.0313 | 0.030829 | 1.02 |
| KLK-0431 | 3444 | 70 | 63.70 | 30.62 | 1 | 0.0206 | 0.020325 | 1.01 |
| KLK-0082 | 12 219 | 41 | 15.99 | 0.00 | 0 | 0.0034 | 0.003355 | 1.01 |
| KLK-0229 | 3492 | 69 | 191.13 | 116.59 | 2 | 0.0200 | 0.019759 | 1.01 |
| KLK-0070 | 2924 | 52 | 188.76 | 133.73 | 1 | 0.0180 | 0.017784 | 1.01 |
| KLK-0078 | 8917 | 67 | 24.79 | 0.00 | 0 | 0.0076 | 0.007514 | 1.01 |
| KLK-0372 | 1000 | 89 | 64.97 | 160.87 | 3 | 0.0899 | 0.089000 | 1.01 |

**`cpc` — 6 linhas divergentes:**

| campaign_id | impressions | clicks | spend | revenue | conversions | cpc informado | cpc recalculado | razão inf./recalc. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KLK-0259 | 1252 | 20 | 95 811.69 | 72.62 | 1 | 0.27 | 4790.5845 | **0.000056** |
| KLK-0109 | 3701 | 205 | 50 544.77 | 63.87 | 8 | 1.18 | 246.5599 | **0.0048** |
| KLK-0224 | 117 801 | 17 670 | 85 023.80 | 0.00 | 170 | 1.22 | 4.8118 | 0.2535 |
| KLK-0100 | 13 342 | 951 | 370.89 | 1162.54 | 27 | 20.00 | 0.3900 | **51.28** |
| KLK-0416 | 32 532 | 1538 | 1138.12 | 2510.55 | 20 | 26.05 | 0.7400 | **35.20** |
| KLK-0240 | 135 857 | 2825 | 2768.50 | 11 936.58 | 88 | 24.17 | 0.9800 | **24.66** |

**`roas` — 27 linhas divergentes** (as duas mais extremas; as outras 25 têm razão entre 0.94 e 1.08 — ver notebook para a lista completa):

| campaign_id | impressions | clicks | spend | revenue | conversions | roas informado | roas recalculado | razão inf./recalc. |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KLK-0259 | 1252 | 20 | 95 811.69 | 72.62 | 1 | 13.45 | 0.000758 | **17745.35** |
| KLK-0109 | 3701 | 205 | 50 544.77 | 63.87 | 8 | 0.26 | 0.001264 | **205.76** |
| (demais 25 linhas) | — | — | — | — | — | — | — | 0.94–1.08 |

**Conclusão 3.1:** a maioria das divergências (12/16 em `ctr`, 25/27 em `roas`, nenhuma em `cpc`) tem razão informado/recalculado entre 0.94 e 1.08 — compatível com **arredondamento** do valor informado, que gera erro relativo > 1% apenas porque a métrica em si é pequena. Mas há dois padrões de **dado corrompido**, não arredondamento:

- **4 linhas com `impressions` entre 12.6 e 14.1 milhões** (`KLK-0148`, `KLK-0232`, `KLK-0258`, `KLK-0413`), razão 148×–1077×. A 5ª maior `impressions` do dataset é 1.57 milhão (ver 3.2) — essas 4 linhas estão 1–2 ordens de magnitude acima de qualquer outro valor, sugerindo dígito(s) extra(s) em `impressions`.
- **6 linhas com `cpc` muito divergente** (`KLK-0100`, `KLK-0109`, `KLK-0224`, `KLK-0240`, `KLK-0259`, `KLK-0416`), razão entre 0.000056× e 51×. Em 3 delas (`KLK-0109`, `KLK-0224`, `KLK-0259`) o `spend` está entre os 8 maiores do dataset, mas `clicks` é baixo demais para esse gasto (ex.: `KLK-0259`: spend=95 811.69 para só 20 clicks), enquanto outras linhas com `spend` igualmente alto têm `cpc` perfeitamente consistente (ver 3.2) — sugere `clicks` ou `spend` trocado/corrompido nessas linhas específicas.

### 3.2 Top 10 `cpc` e top 10 `impressions`

**Top 10 `cpc` informado:**

| campaign_id | cpc | spend | clicks | cpc recalculado |
| :--- | ---: | ---: | ---: | ---: |
| KLK-0416 | 26.05 | 1138.12 | 1538 | 0.74 |
| KLK-0240 | 24.17 | 2768.50 | 2825 | 0.98 |
| KLK-0100 | 20.00 | 370.89 | 951 | 0.39 |
| KLK-0486 | 3.94 | 3707.54 | 941 | 3.94 |
| KLK-0442 | 3.81 | 567.69 | 149 | 3.81 |
| KLK-0012 | 3.78 | 20 521.62 | 5429 | 3.78 |
| KLK-0070 | 3.63 | 188.76 | 52 | 3.63 |
| KLK-0141 | 3.57 | 4494.63 | 1259 | 3.57 |
| KLK-0002 | 3.54 | 1745.22 | 493 | 3.54 |
| KLK-0243 | 3.47 | 8081.63 | 2329 | 3.47 |

Os 3 primeiros (maior `cpc` informado) são exatamente os 3 piores casos do item 3.1 — `cpc` informado muito acima do recalculado. Do 4º ao 10º lugar, `cpc` informado e recalculado **coincidem exatamente**, confirmando que a corrupção é pontual (3 linhas), não um padrão geral em valores altos de `cpc`.

**Top 10 `impressions`:**

| campaign_id | impressions | clicks | ctr informado | ctr recalculado |
| :--- | ---: | ---: | ---: | ---: |
| KLK-0413 | 14 052 761 | 2695 | 0.0374 | 0.000192 |
| KLK-0148 | 13 007 321 | 629 | 0.0148 | 0.000048 |
| KLK-0258 | 12 873 562 | 3704 | 0.0426 | 0.000288 |
| KLK-0232 | 12 594 257 | 345 | 0.0295 | 0.000027 |
| KLK-0435 | 1 570 242 | 72 545 | 0.0462 | 0.046200 |
| KLK-0287 | 903 286 | 43 448 | 0.0481 | 0.048100 |
| KLK-0487 | 426 466 | 63 969 | 0.1500 | 0.149998 |
| KLK-0089 | 360 448 | 19 031 | 0.0528 | 0.052798 |
| KLK-0022 | 334 928 | 17 617 | 0.0526 | 0.052599 |
| KLK-0397 | 328 052 | 13 581 | 0.0414 | 0.041399 |

Confirma o salto de magnitude: as 4 maiores `impressions` (12.6–14.1 milhões) formam um grupo isolado, **~8× maior** que a 5ª colocada (1.57 milhão) — e são exatamente as 4 linhas com `ctr` divergente do item 3.1. Da 5ª em diante, `ctr` informado e recalculado coincidem.

### 3.3 Reconstrução exata de nulos (`cpc` e `revenue`)

**`cpc` nulo (28 linhas):** `spend` disponível em 28/28, `clicks` disponível em 28/28, `clicks == 0` em 0/28 → **28/28 (100%) reconstruíveis** via `cpc = spend/clicks` (lista completa de valores reconstruídos em `notebooks/01_eda.ipynb`, seção 9.3).

**`revenue` nulo (25 linhas):** `roas` disponível em 25/25, `spend` disponível em 25/25 → `revenue = roas * spend` é computável para as 25, mas o `roas` é pouco confiável em 6 delas:

| campaign_id | roas | spend | conversions | problema |
| :--- | ---: | ---: | ---: | :--- |
| KLK-0059 | 0.00 | 237.12 | 3 | roas=0 com conversions>0 (inconsistente) |
| KLK-0110 | 0.00 | 1478.23 | 5 | roas=0 com conversions>0 (inconsistente) |
| KLK-0280 | 0.00 | 23.65 | 0 | roas=0 com conversions=0 (consistente, mas reconstrução trivial = 0) |
| KLK-0466 | 0.00 | 53.69 | 0 | roas=0 com conversions=0 (consistente, mas reconstrução trivial = 0) |
| KLK-0358 | 8.17 | 636.64 | 39 | roas fora do limite IQR geral (limite superior 7.48), dentro do máximo observado (39.92) |
| KLK-0386 | 11.61 | 2616.12 | 614 | roas fora do limite IQR geral (limite superior 7.48), dentro do máximo observado (39.92) |

**Conclusão 3.3:** `cpc` é 100% reconstruível com confiança. Para `revenue`, **19/25 (76%) são reconstruíveis com confiança direta** via `roas*spend`; as outras 6 precisam de critério manual — as 4 com `roas`=0 porque é ambíguo se é um valor real ou um placeholder de dado faltante, e as 2 com `roas` acima do limite IQR porque, embora dentro do máximo histórico (prováveis outliers legítimos), convém confirmar antes de usar a reconstrução direta.

### 3.4 `revenue == 0` por `objective` e por `category`

| objective | revenue>0 | revenue==0 | total | % zero |
| :--- | ---: | ---: | ---: | ---: |
| app_install | 44 | 0 | 44 | 0.0% |
| awareness | 24 | 33 | 57 | **57.9%** |
| conversions | 152 | 4 | 156 | 2.6% |
| engagement | 46 | 31 | 77 | **40.3%** |
| traffic | 137 | 4 | 141 | 2.8% |

| category | revenue>0 | revenue==0 | total | % zero |
| :--- | ---: | ---: | ---: | ---: |
| App Install | 76 | 0 | 76 | 0.0% |
| Branding | 0 | 61 | 61 | **100.0%** |
| E-commerce | 154 | 2 | 156 | 1.3% |
| Lead Gen | 88 | 3 | 91 | 3.3% |
| SaaS | 85 | 6 | 91 | 6.6% |

**Conclusão 3.4:** sim, `revenue == 0` se concentra fortemente em `objective` ∈ {`awareness` (57.9%), `engagement` (40.3%)} e quase exclusivamente em `category` = `Branding` (100.0%, 61/61). Nos demais, a taxa é baixa (1.3%–6.6%, exceto `app_install`=0%). Isso é esperado do ponto de vista de negócio — campanhas de `awareness`/`engagement` e da categoria `Branding` não visam conversão direta — e **não indica erro de dado**.

### 3.5 Sobreposição de nulos entre colunas

Colunas analisadas: `has_subtitle`, `video_duration_s`, `music_voice_ratio`, `cpc`, `revenue`, `avg_watch_time_s`, `engagement_rate`.

Nº de nulos simultâneos por linha: **0 nulos → 309/500 (61.8%)**; **1 nulo → 152/500 (30.4%)**; **2 nulos → 34/500 (6.8%)**; **3 nulos → 5/500 (1.0%)**; 4+ nulos → 0.

Razão observado/esperado (sob independência) para os pares fora da diagonal — todas as contagens absolutas são pequenas (≤ 6 linhas):

| Par | Observado | Esperado | Razão |
| :--- | ---: | ---: | ---: |
| `music_voice_ratio` × `engagement_rate` | 6 | 2.43 | 2.47 |
| `music_voice_ratio` × `avg_watch_time_s` | 4 | 2.05 | 1.95 |
| `has_subtitle` × `engagement_rate` | 4 | 2.94 | 1.36 |
| `video_duration_s` × `revenue` | 3 | 1.95 | 1.54 |
| `cpc` × `avg_watch_time_s` | 0 | 1.51 | 0.00 |
| `cpc` × `engagement_rate` | 0 | 1.79 | 0.00 |
| (demais pares) | — | — | 0.46–1.42 |

**Conclusão 3.5:** não há nenhum par com sobreposição sistemática e expressiva — todas as razões ficam entre 0 e 2.47, com contagens absolutas pequenas (máximo 6 linhas). A missingness entre as colunas parece **independente** (cada coluna tem seu próprio mecanismo de nulo), reforçando a conclusão da seção 1.1 de que os nulos são majoritariamente MCAR/MAR e podem ser imputados coluna a coluna, sem exigir um tratamento conjunto.

---

## 4. Decisões finais

Correções determinísticas, linha a linha, aplicadas **antes do split** (não dependem de nenhuma partição treino/teste, só de relações entre colunas da própria linha); a imputação estatística de nulos sem correção determinística possível continua dentro do `Pipeline`/`ColumnTransformer` na Parte 2 (regra c do CLAUDE.md — evitar vazamento de dados).

**Regra geral de triangulação:** a fonte de verdade é o valor **consistente com mais relações entre colunas**, não automaticamente a coluna bruta. Nos casos de dado corrompido encontrados aqui, isso significou usar `ctr`/`cpc`/`roas` informados (e a relação entre eles) para corrigir `impressions`/`spend`, em vez de confiar ciegamente nessas duas colunas.

Implementado em `klike.data.clean()` (pura — recebe o df de `load_raw()` e retorna uma cópia corrigida) e `klike.data.load_clean()` (`load_raw()` + `clean()`). Resultado salvo em `data/processed/klike_clean.parquet`. **A partir daqui, todas as análises usam `load_clean()`.**

| Problema | Linhas | Tratamento final | Justificativa | Onde é aplicado |
| :--- | :--- | :--- | :--- | :--- |
| `impressions` corrompido | `KLK-0148`, `KLK-0232`, `KLK-0258`, `KLK-0413` (4) | `impressions = round(clicks / ctr_informado)` | `clicks` e `ctr` informados são plausíveis e consistentes com o resto do dataset; `impressions` bruto está 1–2 ordens de magnitude acima de qualquer outra linha | `clean()` |
| `spend` corrompido | `KLK-0259`, `KLK-0109`, `KLK-0224` (3) | `spend = cpc_informado * clicks` | `cpc` e `roas` informados concordam entre si (triangulação); confirmado por `revenue_informado/spend_corrigido ≈ roas_informado` com erro de 0.014% (`KLK-0259`) e 1.55% (`KLK-0109`), ambos < 2%. `KLK-0224` (`revenue`=0) não é verificável por essa via — limitação registrada em `correcao_aplicada` | `clean()` |
| `cpc` informado corrompido | `KLK-0100`, `KLK-0416`, `KLK-0240` (3) | Substituído pelo recálculo `spend/clicks` | `spend` e `clicks` são plausíveis; só o `cpc` informado estava errado | `clean()` (passo 5, recálculo global) |
| `cpc` nulo | 28 linhas (inclui `KLK-0386`, que também tinha `revenue` nulo) | Reconstruído = `spend/clicks` (recálculo global) | `spend` e `clicks` disponíveis e consistentes em 100% dessas linhas (seção 3.3) | `clean()` (passo 5) |
| `revenue` nulo | 25 linhas | Reconstruído = `roas_informado * spend` (usando o `spend` já corrigido) | Outlier de `roas` não é critério de dado ruim — reconstrução direta para as 25, incluindo `KLK-0358`/`KLK-0386` (roas fora do IQR geral) | `clean()` |
| `revenue` reconstruído ambíguo | `KLK-0059`, `KLK-0110` (`roas`=0 com `conversions`>0) | Mantido `revenue`=0 (mesmo critério), mas marcado com nota em `correcao_aplicada` | A inconsistência entre `roas` informado e `conversions`>0 não é resolvível com os dados disponíveis; melhor registrar a incerteza do que inventar um valor | `clean()` (flag) |
| Ruído de arredondamento em `ctr`/`cpc`/`roas` | Todas as 500 linhas | Recalculado `ctr = clicks/impressions`, `cpc = spend/clicks`, `roas = revenue/spend` a partir das colunas brutas já corrigidas | Elimina o ruído de arredondamento do valor informado e garante consistência interna; validado: 0 violações de faixa e 0 divergências > 1% no dado limpo (seção 10.2 do notebook) | `clean()` (passo 5) |
| Nulos sem correção determinística: `has_subtitle` (46), `video_duration_s` (39), `music_voice_ratio` (38), `avg_watch_time_s` (27), `engagement_rate` (32) | 182 ocorrências de nulo, 152 linhas distintas (com sobreposição entre colunas — ver seção 3.5) | **Não tratados em `clean()`** — ficam para imputação estatística + indicador de nulo | Não há relação exata com outras colunas que permita reconstrução determinística; regra (c) do CLAUDE.md exige que a imputação estatística fique dentro do Pipeline, nunca aplicada manualmente antes do split/CV | `Pipeline` (Parte 2) |

### Notas adicionais

- **ROAS/revenue por categoria:** análises de `roas`/`revenue` devem **excluir a categoria `Branding`** (`revenue`=0 em 100% das linhas — ver seção 3.4 — é um **zero estrutural**, não um dado a modelar como variação de performance) e devem tratar `revenue`=0 em `objective` ∈ {`awareness`, `engagement`} como **esperado** (57.9%–40.3% de zeros nesses grupos, também estrutural, não erro).
- **Regra de fallback do Recommendations Engine para grupos pequenos:** ao estimar um efeito contextual, usar o nível mais específico com `n ≥ 20`: tentar `platform × category`; se `n < 20`, cair para `platform`; se ainda `n < 20` (não deve ocorrer, já que toda `platform` tem n ≥ 83), cair para o nível global. **Sempre reportar o nível e o `n` usados na recomendação** (regra e do CLAUDE.md).
- **Imputação no Pipeline (Parte 2):** `has_subtitle` → `most_frequent` + `add_indicator`; `video_duration_s` e `music_voice_ratio` → mediana + `add_indicator`. (`avg_watch_time_s` e `engagement_rate` seguem a definir na Parte 2.)
