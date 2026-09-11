"""Single entry point. `python -m src.cli reproduce` regenerates headline numbers."""
from __future__ import annotations

import json

import typer
from rich.console import Console

from src import config, llm, pipeline, sampling, taxonomy
from src import ingest as ing

app = typer.Typer(add_completion=False, help="BA support agent pipeline")
con = Console()


@app.command()
def ingest():
    """Build data/interim/ba_pairs.parquet from the raw CSV."""
    con.print(f"wrote {ing.run_ingest()}")


@app.command("taxonomy")
def taxonomy_propose(clusters: int = 12):
    """Cluster corpus messages and dump data/interim/clusters.json for naming."""
    taxonomy.propose(n_clusters=clusters)
    con.print("wrote data/interim/clusters.json - now hand-write taxonomy.json")


@app.command()
def pool(n: int = config.GOLDEN_TARGET):
    """Build the stratified golden pool to label."""
    p = sampling.build_golden_pool(n)
    con.print(f"pooled {len(p)} examples")


@app.command()
def run(threshold: float = 0.5):
    """Run the agent over the labelled golden set (round 1) and print a
    preview - does not write reports/results.json (use `evaluate` for that)."""
    import pandas as pd

    golden = pd.read_json(config.GOLDEN_DIR / "golden.jsonl", lines=True)
    golden = golden[golden["round"] == 1].reset_index(drop=True)
    out = pipeline.run_agent(golden, threshold=threshold)
    con.print(out.head(10))
    con.print(f"[dim]{len(out)} rows[/dim]")


@app.command()
def evaluate(k: float = 20.0):
    """Run the agent over the golden set and write reports/results.json."""
    res = pipeline.run_all(k=k)
    con.print_json(json.dumps(res, default=float))


@app.command()
def reproduce():
    """Reproduce headline results from the committed cache. No API key needed."""
    stats = llm.cache_stats()
    con.print(f"[dim]cache entries: {stats['entries']}[/dim]")
    res = pipeline.run_all(k=20.0)
    d = res["deployment"]
    i = res["intent"]["agent"]
    con.print("\n[bold]HEADLINE[/bold]")
    con.print(f"  intent accuracy : {i['accuracy']:.3f} "
              f"[{i['accuracy_ci'][0]:.3f}, {i['accuracy_ci'][1]:.3f}]")
    con.print(f"  macro-F1        : {i['macro_f1']:.3f}")
    con.print(f"  auto-handled    : {d['coverage']:.1%} of volume")
    con.print(f"  harm rate       : {d['harm_rate']:.1%} among auto-handled")
    con.print(f"  at threshold    : {d['threshold']:.2f} (cost ratio k={d['k']})")


if __name__ == "__main__":
    app()
