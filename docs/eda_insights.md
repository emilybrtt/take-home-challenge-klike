# Insights — Parte 1, perguntas 2-4

Análise de atributos acionáveis x performance (pergunta 2), padrões por plataforma (pergunta 3) e proposta de visualizações (pergunta 4). Código completo em `notebooks/01_eda.ipynb`, seção 11 ("Parte 1 — Insights"), usando sempre `load_clean()`. Dados são **observacionais** — todos os efeitos são reportados como "associado a", nunca "causa"/"aumenta".

**Metodologia:** mediana de cada nível vs. um nível de referência fixo por atributo (binários: "sem"; `format`: `vertical`; `text_density`: `low`; `video_duration_bin`: `<=15s`; `music_voice_bin`: `mais_voz`), com `n` de cada grupo. Efeito = lift relativo (%) para a maioria das métricas, exceto `klike_score` (diferença absoluta em pontos). IC95% por bootstrap de duas amostras independentes (1000 reamostragens, `random_state=42`, cada grupo reamostrado separadamente com o próprio `n`). Evidência = IC não cruza 0. **212 comparações no total** (72 gerais + 140 por plataforma) → **~10.6 falsos positivos esperados só pelo acaso** (IC95%). Encontramos 51 achados com evidência — bem mais que o esperado ao acaso, mas os "isolados" (abaixo) devem ser lidos com mais cautela que os "robustos".

## Resumo — achados com evidência (robustos primeiro)

**Robustos** (IC não cruza 0 E repete com o mesmo sinal em ≥1 outra métrica ou em ≥2 plataformas):

1. **`has_hook`** — maior efeito geral: ctr +67.6%, engagement_rate +36.5%, klike_score +17.3 pts, watch_rate +34.3% (n=256); repete nas 3 plataformas (Meta/TikTok/LinkedIn).
2. **`has_face`** — ctr +37.0%, engagement_rate +54.7%, klike_score +11.4 pts, watch_rate +16.3% (n=269); repete em Meta, TikTok e LinkedIn.
3. **`has_cta`** — klike_score +7.3 pts (n=347), repete em Meta/TikTok/LinkedIn; **sem** evidência em ctr/engagement_rate/watch_rate/cvr/roas gerais.
4. **`has_subtitle`** — klike_score +5.7 pts (n=263), repete em Meta/TikTok; engagement_rate +79% em LinkedIn (n=42, amostra menor).
5. **`text_density=high`** (vs. `low`) — ctr -36.7% (n=91), klike_score -8.8 pts (n=91), roas -41.5% (n=64); repete em Meta.
6. **`format=horizontal`** (vs. `vertical`) — engagement_rate -40.5% (n=104), roas -57.2% (n=84); repete em Meta. **Cautela:** TikTok tem 0% de `horizontal` (seção 11.6) — a comparação é, em parte, Meta/LinkedIn vs. TikTok.
7. **`video_duration_bin=>30s`** (vs. `<=15s`) — ctr -27.3% (n=56), engagement_rate -26.3% (n=55), roas -35.5% (n=40); repete em Meta (ctr).

**Isolados** (evidência numa única combinação, sem repetição — 7 achados): `format=quadrado` x engagement_rate (-34.2%, n=109); `text_density=medium` x klike_score (+7.5 pts, n=230); `video_duration_bin=15-30s` x roas (-38.0%, n=122); `music_voice_bin=equilibrado` x watch_rate (+12.5%, n=196, repete só no TikTok); `music_voice_bin=mais_musica` x watch_rate no Meta (+25.0%, n=22).

**Confundimento por plataforma (item 11.9):** nenhum efeito muda de sinal entre o geral e nenhuma plataforma — **não há paradoxo de Simpson clássico**. 2 achados gerais (`format=quadrado` e `video_duration_bin=>30s`, ambos x engagement_rate) não se confirmam em nenhuma plataforma individualmente; como o sinal não se inverte, a explicação mais simples é perda de poder estatístico (n menor por plataforma), não confundimento — por isso ficam classificados "isolado".

**Score x performance real:** `klike_score` correlaciona (Spearman) positiva e significativamente (p<0.05) com TODAS as métricas reais: ctr (rho=0.538, n=500), watch_rate (rho=0.380, n=437), engagement_rate (rho=0.295, n=468), roas (rho=0.218, n=361), cvr (rho=0.167, n=500) — o score está alinhado com o resultado real, mais fortemente com cliques que com conversão/receita.

## Achados sem evidência (pareciam relevantes, mas o IC cruza 0)

- `has_subtitle` x ctr/engagement_rate/watch_rate/cvr/roas (geral) — todas sem evidência (só `klike_score` tem).
- `has_cta` x ctr/engagement_rate/watch_rate/cvr/roas (geral) — todas sem evidência (só `klike_score` tem).
- `music_voice_bin` (`equilibrado`/`mais_musica` vs. `mais_voz`) x ctr/engagement_rate/klike_score/cvr/roas — nenhuma evidência (só uma combinação de `watch_rate`, isolada).
- `video_duration_bin` x `watch_rate`, em TODAS as faixas — nenhuma evidência: o efeito "mecânico" esperado (vídeo mais longo → fração assistida menor) **não se confirma** na fração assistida, só nos segundos absolutos (seção 11.10: mediana de `avg_watch_time_s` sobe de 5.0s para 12.5s para 21.2s por faixa, mas `watch_rate` não difere com evidência).
- `format=quadrado` x roas: -25.3%, mas IC [-48.7%, +9.4%] cruza 0.
- `text_density=medium` x ctr/engagement_rate/watch_rate/cvr/roas — nenhuma evidência (só `klike_score`, e isolada).
- Correlações contínuas: a maioria de `video_duration_s`/`music_voice_ratio` x métricas não é significativa, exceto `video_duration_s` x engagement_rate (rho=-0.131, p=0.006) e x roas (rho=-0.170, p=0.002), e `music_voice_ratio` x watch_rate (rho=0.134, p=0.007) — correlações fracas, mas estatisticamente presentes.

## Proposta de 4 visualizações (para o time de marketing)

| # | Pergunta que responde | Tipo de gráfico | Por quê |
| :-- | :-- | :-- | :-- |
| 1 | Quais atributos de criativo têm o efeito mais forte e confiável? | **Forest plot** (ponto = lift, barra = IC95%) por atributo, ordenado por força do efeito, cor para robusto vs. isolado | Mostra o efeito E a incerteza juntos — deixa visualmente claro quais atributos têm evidência sólida (IC não cruza 0, estreito) vs. quais são "quase lá" (IC cruza 0), sem exigir que o time entenda bootstrap |
| 2 | O que funciona é igual nas 3 plataformas? | **Heatmap** atributo x plataforma, cor = sinal/magnitude do lift, célula hachurada/cinza quando `n<20` (amostra insuficiente) | Comparação lado a lado imediata; a hachura comunica "não confie nesse número" sem precisar de texto |
| 3 | O score da Klike reflete o resultado real da campanha? | **Grade de scatter plots** (klike_score no eixo x, uma métrica por painel: ctr, watch_rate, engagement_rate, roas) com linha de tendência | Mais intuitivo que reportar "rho de Spearman" para um público não-técnico; a dispersão em torno da linha já comunica a força da relação |
| 4 | O mix de criativos é diferente por plataforma? | **Barras 100% empilhadas** de `format` e de `video_duration_bin` por plataforma | Explica achados como "horizontal tem roas pior" com uma ressalva visual imediata: TikTok simplesmente não usa horizontal — é mix, não (só) o atributo |

Visualizações finais (código/estilo) ainda não geradas — aguardando aprovação.
