## 📋 Contexto

Você recebeu um dataset (`klike_challenge_dataset.csv`) contendo **500 registros de campanhas de vídeo** veiculadas em diferentes plataformas. Cada registro representa um criativo de vídeo com seus atributos e métricas de performance.

O **Klike Score** (coluna `klike_score`) é uma pontuação de 0 a 100 que representa a qualidade geral do criativo, calculada por nosso sistema de IA com base em múltiplos fatores.

### Dicionário de dados

| Coluna | Descrição |
| :---- | :---- |
| `campaign_id` | Identificador único da campanha |
| `date` | Data de veiculação |
| `platform` | Plataforma (Meta, TikTok, LinkedIn) |
| `category` | Categoria do anunciante (E-commerce, SaaS, App Install, Branding, Lead Gen) |
| `objective` | Objetivo da campanha (awareness, traffic, conversions, engagement, app\_install) |
| `target_audience_age` | Faixa etária do público-alvo (18-24, 25-34, 35-44, 45+) |
| `is_retargeting` | Se a campanha é de retargeting (True) ou prospecção (False) |
| `video_duration_s` | Duração do vídeo em segundos |
| `format` | Formato do vídeo (vertical, horizontal, quadrado) |
| `has_subtitle` | Se o vídeo possui legendas |
| `has_cta` | Se o vídeo possui call-to-action |
| `has_hook` | Se o vídeo possui hook nos primeiros 3 segundos |
| `has_face` | Se o vídeo apresenta rosto humano |
| `text_density` | Quantidade de texto on-screen (low, medium, high) |
| `music_voice_ratio` | Proporção música/voz (0 \= só voz, 1 \= só música) |
| `impressions` | Número de impressões |
| `clicks` | Número de cliques |
| `ctr` | Click-through rate |
| `cpc` | Custo por clique (R\$) |
| `spend` | Gasto total (R\$) |
| `conversions` | Número de conversões |
| `revenue` | Receita gerada (R\$) |
| `roas` | Return on Ad Spend |
| `avg_watch_time_s` | Tempo médio de visualização do vídeo (segundos) |
| `engagement_rate` | Taxa de engajamento (likes \+ comentários \+ compartilhamentos / impressões) |
| `klike_score` | Score de qualidade do criativo (0-100) — **target** |

---

## 🧩 O Desafio

### Parte 1 — Análise Exploratória (EDA)

Explore o dataset e responda:

1. **Qual o perfil geral dos dados?** Distribuições, valores faltantes, outliers — como você trataria cada caso?  
2. **Quais atributos do criativo mais se correlacionam com boas métricas de performance?** (CTR, ROAS, conversões, etc.)  
3. **Existem padrões diferentes por plataforma?** O que funciona no TikTok funciona no LinkedIn?  
4. **Crie pelo menos 3 visualizações** que você usaria para apresentar insights a um time de marketing.

# Brainstorm
## Perguntas
- o score é calculado antes ou depois da veiculação?
klike_score -> mede a “qualidade do criativo”
    valor = pontuar o vídeo antes de gastar dinheiro com ele, quando ainda não há CTR, ROAS, cliques. 
    Se usarmos essas métricas de performance como features, o modelo pode até ir bem, mas fica inútil para esse uso.

Solução: treinar dos dois jeitos e comparar