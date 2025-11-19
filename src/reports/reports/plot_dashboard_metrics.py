"""Plot helper that turns dashboard KPI CSVs into PNG charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_dashboard(csv_path: Path, output_path: Path) -> None:
    """Render a KPI overview plot and save it to disk."""
    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV vazio, nada para plotar.")
        return

    df["generated_at"] = pd.to_datetime(df["generated_at"])
    df = df.sort_values("generated_at")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(
        df["generated_at"],
        df["alerts_total"],
        label="Alertas",
        color="#1f77b4",
        linewidth=2,
    )
    ax1.plot(
        df["generated_at"],
        df["dispatch_total"],
        label="Despachos",
        color="#ff7f0e",
        linewidth=2,
    )
    ax1.set_ylabel("Contagem acumulada")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        df["generated_at"],
        df["coverage_rate"],
        label="Cobertura (%)",
        color="#2ca02c",
        linestyle="--",
    )
    ax2.plot(
        df["generated_at"],
        df["mean_response_steps"],
        label="Resp. média (passos)",
        color="#d62728",
        linestyle=":",
    )
    ax2.set_ylabel("Cobertura / Resposta")
    ax2.legend(loc="upper right")

    plt.title("KPIs do Dashboard (por hora in-game)")
    fig.autofmt_xdate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico gravado em {output_path}")


def main() -> None:
    """Parse CLI arguments and generate the KPI plot."""
    parser = argparse.ArgumentParser(description="Gera um gráfico a partir do dashboard_metrics.csv.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("interface/reports/dashboard_metrics.csv"),
        help="CSV de entrada (default: interface/reports/dashboard_metrics.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dashboard_metrics.png"),
        help="Imagem de saída (default: interface/reports/dashboard_metrics.png).",
    )
    args = parser.parse_args()
    plot_dashboard(args.csv, args.output)


if __name__ == "__main__":
    main()
