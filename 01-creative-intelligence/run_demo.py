"""
Demo runner — test the full pipeline without the MCP server.
Usage: python run_demo.py
"""

import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data.synthetic import generate_campaign_data
from ml.classifier import CreativeClassifier
from ml.anomaly import AnomalyDetector
from agents.orchestrator import OrchestratorAgent

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]Ad Tech Intelligence System[/bold cyan]\n"
        "Hierarchical Multi-Agent Pipeline with ML",
        border_style="cyan",
    ))

    # ── Data ───────────────────────────────────────────────────────────────
    console.print("\n[yellow]Generating synthetic campaign data...[/yellow]")
    data = generate_campaign_data(n_creatives=30, n_days=25)
    console.print(
        f"  ✓ {len(data['creatives'])} creatives  "
        f"{len(data['performance'])} records  "
        f"{len(data['segments'])} segments"
    )

    # ── ML models ──────────────────────────────────────────────────────────
    console.print("\n[yellow]Training ML models...[/yellow]")
    clf = CreativeClassifier()
    tr = clf.train(data["performance"])
    console.print(f"  ✓ Random Forest: {tr['cv_accuracy']:.1%} CV accuracy")
    console.print(f"    Top feature: {next(iter(tr['feature_importances']))}")

    anom = AnomalyDetector()
    anom.fit_performance_detector(data["performance"])
    anom.fit_bid_detector(data["bid_landscape"])
    console.print("  ✓ Isolation Forest: 2 detectors fitted")

    # ── Orchestrator ───────────────────────────────────────────────────────
    console.print("\n[yellow]Initialising agent hierarchy...[/yellow]")
    orch = OrchestratorAgent(data, {"classifier": clf, "anomaly_detector": anom})

    agents_table = Table(show_header=False, box=None)
    for name in orch.agents:
        agents_table.add_row(f"  ✓ {name}")
    console.print(agents_table)

    # ── Run pipeline ───────────────────────────────────────────────────────
    console.print("\n[bold green]Running full pipeline...[/bold green]")
    result = orch.run_full_analysis(campaign_id="CAMP_001")

    console.print("\n" + "═" * 64)
    console.print("[bold]ORCHESTRATOR RESULTS[/bold]")
    console.print("═" * 64)

    if summary := result.get("executive_summary"):
        console.print(Panel(summary, title="Executive Summary", border_style="green"))

    if actions := result.get("priority_actions"):
        console.print("\n[bold]Priority Actions:[/bold]")
        if isinstance(actions, list):
            for i, a in enumerate(actions[:5], 1):
                console.print(f"  {i}. {a}")

    ens = result.get("ensemble_result", {})
    console.print(f"\n[bold]Ensemble Decision:[/bold] {ens.get('final_recommendation', '—')}")
    console.print(f"  Confidence:  {ens.get('confidence', 0):.1%}")
    console.print(f"  Agreement:   {ens.get('agreement_score', 0):.1%}")
    console.print(f"  Quality:     {ens.get('quality_flag', '—')}")

    crit = result.get("critique_summary", {})
    concerns = crit.get("major_concerns", [])
    if concerns:
        console.print(f"\n[bold red]Critic Flags ({len(concerns)} concerns):[/bold red]")
        for c in concerns[:3]:
            console.print(f"  ⚠  {c}")

    console.print(f"\n[dim]Completed in {result.get('execution_time_seconds', '?')}s[/dim]")
    console.print(f"[dim]Agents: {', '.join(result.get('agents_run', []))}[/dim]")

    with open("analysis_output.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    console.print("\n[dim]Full output → analysis_output.json[/dim]")


if __name__ == "__main__":
    main()
