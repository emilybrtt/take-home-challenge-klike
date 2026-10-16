# Klike Data Science Challenge

## Visão geral

## Como rodar

Testado com **Python 3.14**.

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

### 1. Perfil e qualidade dos dados

500 campanhas, 25 colunas. Como tratamos cada problema encontrado:

- **Valores faltantes** — 7 colunas com 5-9% de nulos cada, independentes entre si (não concentrados nas mesmas linhas). `cpc` e `revenue` foram **reconstruídos exatamente** a partir de outras colunas (cliques/gasto e ROAS/gasto). `has_subtitle`, `video_duration_s` e `music_voice_ratio` ficam para **imputação (moda/mediana) + indicador de nulo dentro do Pipeline** de modelagem — nunca antes do split, para não vazar dado. `avg_watch_time_s` e `engagement_rate` são só **excluídos no cálculo que usa aquela métrica específica** (análise de casos disponíveis).
- **Outliers** — mantidos: são extremos legítimos (ex.: campanhas de gasto muito alto), não erro de digitação. As análises usam mediana e, quando relevante, escala log; nenhum outlier legítimo foi removido.
- **Dados inconsistentes** — um bug converteu silenciosamente 46 nulos de `has_subtitle` em "tem legenda" (corrigido). Em 4 campanhas, `impressions` tinha valores fora de escala (1-2 ordens de grandeza acima das demais) e inconsistentes com cliques e CTR informados; em 3 campanhas, o gasto era inconsistente com cliques/receita. Em vez de descartar essas linhas, reconstruímos os valores mais prováveis usando as colunas que concordavam entre si (**triangulação**).
- **Branding** — receita zero em 100% das campanhas (61 de 61). Não é erro: esse tipo de campanha não é medido por conversão/receita. Excluímos `branding` das análises de ROAS.
- Detalhes completos, com os números de cada correção: [`docs/eda_qualidade.md`](docs/eda_qualidade.md).

### 2. Atributos x performance

![Forest plot](reports/figures/forest_ctr_engajamento_score.png)

Hook nos 3 primeiros segundos e rosto humano no vídeo são os atributos com o efeito mais forte e mais consistente: ambos associados a CTR, engajamento e score mais altos, de forma repetida nas 3 plataformas. Vídeo com muito texto na tela e vídeos com mais de 30 segundos aparecem do lado oposto.

![Score x performance](reports/figures/score_x_ctr_e_roas.png)

O score da Klike acompanha o resultado real: campanhas no quintil de score mais alto têm CTR ~3,4x maior e ROAS ~2,3x maior que as do quintil mais baixo.

**Score x performance real.** Dois atributos — **CTA** (repete nas 3 plataformas) e **legendas** (repete em Meta e TikTok) — elevam o `klike_score` de forma consistente, mas **não têm evidência de elevar nenhuma métrica real de performance** (CTR, engajamento, taxa de conversão ou ROAS). Isso importa: o score da Klike, sozinho, pode levar a recomendações que melhoram a nota do criativo sem melhorar o resultado da campanha. Por isso o Recommendations Engine (Parte 3) separa práticas com evidência em performance real das que só têm evidência em score.

![Checklist do criativo](reports/figures/checklist_criativo_ctr.png)

Juntando as práticas com evidência em performance real (hook, rosto humano, pouco texto, vídeo curto) num único "placar": quanto mais boas práticas, maior o CTR mediano. **Ressalva:** essas 4 práticas foram escolhidas a partir desta mesma análise, o que favorece o resultado; a validação em dados não vistos é feita na Parte 2.

### 3. Diferenças entre plataformas

![Heatmap por plataforma](reports/figures/heatmap_ctr_atributo_plataforma.png)

Hook é o único atributo associado a CTR maior nas 3 plataformas ao mesmo tempo. Rosto humano só tem evidência no TikTok; muito texto e vídeo longo, só no Meta. O formato do vídeo não pôde ser comparado no TikTok, que praticamente não usa formato horizontal (mix de plataforma, não falta de efeito).

### 4. Visualizações

As 4 visualizações da pergunta 4 estão acima, cada uma com a pergunta que responde: o que mais importa no criativo (forest plot), o score reflete performance real (score x CTR/ROAS), quantas boas práticas um criativo acumula (checklist) e o que muda por plataforma (heatmap). Um 5º gráfico (mix de formato/duração por plataforma) fica só no notebook técnico, como apoio à leitura do heatmap.

### Nota de método

Todo efeito usa a **mediana** (não a média — as métricas são bem assimétricas) e vem com uma **faixa de incerteza por bootstrap** (1000 reamostragens); só contamos um efeito como achado quando essa faixa não cruza o zero. Testamos 212 combinações de atributo x métrica x plataforma ao todo — ao acaso, ~11 delas dariam "achado" mesmo sem efeito real; encontramos 51, a maioria (44) confirmada em mais de uma métrica ou plataforma. Detalhes de cada achado, incluindo os que pareciam relevantes mas não se confirmaram: [`docs/eda_insights.md`](docs/eda_insights.md).

## Parte 2 — Modelagem

## Parte 3 — Recommendations Engine

## Parte 4 — Visão de produto

## Limitações
