# Klike Data Science Challenge

## Visão geral

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Os notebooks estão em `notebooks/`, numerados na ordem em que devem ser executados:

- `01_eda.ipynb` — Parte 1
- `02_modeling.ipynb` — Parte 2
- `03_recommendations.ipynb` — Parte 3

Funções reutilizáveis ficam em `src/klike/`.

## Parte 1 — EDA

Análise completa, com código, em `notebooks/01_eda.ipynb`.

### Qualidade dos dados

- **Um bug foi corrigido antes de qualquer análise**: a coluna `has_subtitle` tinha 46 valores faltantes que estavam sendo transformados silenciosamente em "tem legenda" por um erro de conversão de tipo. Corrigido, e os valores faltantes parecem majoritariamente aleatórios.
- **4 campanhas tinham o número de impressões inflado por um erro de digitação** (1-2 ordens de magnitude acima de qualquer outra campanha) e **3 campanhas tinham o gasto incompatível com cliques/receita informados**. Em vez de descartar essas linhas, reconstruímos os valores corretos cruzando as colunas que ainda batiam entre si (ex.: cliques e CTR informados concordavam; só o número de impressões estava errado).
- **Campanhas de branding têm receita zero em praticamente 100% dos casos** — não é erro, é porque esse tipo de campanha não é medido por conversão/receita. Excluímos `branding` das análises de ROAS para não distorcer a conta.
- Detalhes completos, com os números de cada correção: [`docs/eda_qualidade.md`](docs/eda_qualidade.md).

### O que mais importa num criativo

![Forest plot](reports/figures/forest_ctr_engajamento_score.png)

Hook nos 3 primeiros segundos e rosto humano no vídeo são os atributos com o efeito mais forte e mais consistente: ambos aparecem associados a CTR, engajamento e score mais altos, de forma repetida nas 3 plataformas. Vídeo com muito texto na tela e vídeos com mais de 30 segundos aparecem do lado oposto — associados a métricas piores.

![Heatmap por plataforma](reports/figures/heatmap_ctr_atributo_plataforma.png)

O que funciona para CTR muda pouco de uma plataforma para outra — a maior diferença é o formato: TikTok simplesmente não usa vídeo horizontal (por isso a célula aparece "sem dados").

![Score x performance](reports/figures/score_x_ctr_e_roas.png)

O score da Klike acompanha o resultado real: campanhas no quintil de score mais alto têm CTR ~3,4x maior e ROAS ~2,3x maior que as do quintil mais baixo.

![Checklist do criativo](reports/figures/checklist_criativo_ctr.png)

Juntando as práticas com evidência de impacto real (hook, rosto humano, pouco texto, vídeo curto) num único "placar": criativos com as 4 têm CTR quase 4x maior que os sem nenhuma — o efeito se acumula.

### Score x performance real

Dois atributos — **CTA** e **legendas** — elevam o `klike_score` de forma consistente (repetem nas 3 plataformas), mas **não têm evidência de elevar nenhuma métrica real de performance** (CTR, engajamento, taxa de conversão ou ROAS). Isso importa porque significa que o score da Klike, sozinho, pode levar a recomendações que melhoram a nota do criativo sem melhorar o resultado da campanha. Por isso, o Recommendations Engine (Parte 3) separa as práticas com evidência em performance real das que só têm evidência em score.

### Nota de método

Todo efeito usa a **mediana** (não a média — as métricas são bem assimétricas) e vem com uma **faixa de incerteza por bootstrap** (1000 reamostragens); só contamos um efeito como achado quando essa faixa não cruza o zero. Testamos 212 combinações de atributo x métrica x plataforma ao todo — ao acaso, ~11 delas dariam "achado" mesmo sem efeito real; encontramos 51, a maioria (44) confirmada em mais de uma métrica ou plataforma. Detalhes de cada achado, incluindo os que pareciam relevantes mas não se confirmaram: [`docs/eda_insights.md`](docs/eda_insights.md).

## Parte 2 — Modelagem

## Parte 3 — Recommendations Engine

## Parte 4 — Visão de produto

## Limitações
