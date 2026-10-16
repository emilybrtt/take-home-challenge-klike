"""CLI do Recommendations Engine (Parte 3).

Uso:
    python -m klike.cli --campaign-id KLK-0001
    python -m klike.cli --campaign-id KLK-0001 --json
"""

from __future__ import annotations

import argparse
import json
import sys

from klike import recommender as rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recommendations Engine — Klike")
    parser.add_argument("--campaign-id", required=True, help="campaign_id do dataset limpo (ex.: KLK-0001)")
    parser.add_argument("--json", action="store_true", help="imprime a saída estruturada (JSON) em vez do texto")
    parser.add_argument("--n-min", type=int, default=3, help="nº mínimo de recomendações (padrão: 3)")
    args = parser.parse_args(argv)

    df = rec.carregar_dados()
    linhas = df.loc[df["campaign_id"] == args.campaign_id]
    if linhas.empty:
        print(f"campaign_id '{args.campaign_id}' não encontrado em data/processed/klike_clean.parquet", file=sys.stderr)
        return 1
    campanha = linhas.iloc[0]

    score_atual = rec.prever_score(campanha)
    recomendacoes = rec.recomendar(campanha, n_min=args.n_min)

    if args.json:
        saida = {
            "campaign_id": args.campaign_id,
            "platform": campanha["platform"],
            "score_previsto_atual": round(score_atual, 1),
            "recomendacoes": recomendacoes,
        }
        print(json.dumps(saida, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"Campanha {args.campaign_id} ({campanha['platform']} / {campanha['category']})")
    print(f"Score Klike previsto atual: {score_atual:.1f}\n")
    print("Recomendações:")
    for i, r in enumerate(recomendacoes, start=1):
        print(f"{i}. {r['texto']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
