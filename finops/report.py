"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart."""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 reasoning: dict | None = None, carbon: dict | None = None,
                 analysis: dict | None = None) -> str:
    """Return a markdown cost-optimization report."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) |",
        "|---|---|",
    ]
    for name, amount in levers.items():
        lines.append(f"| {name} | ${amount:,.0f} |")
    if analysis:
        lines += [
            "",
            "## The GPU-Util Lie",
            "",
            "`nvidia-smi` GPU-Util only reports that *some* kernel is resident during the sample",
            "window — it measures clock activity, not FLOP throughput. An LLM decode phase is",
            "memory-bound (arithmetic intensity ~1-2 FLOP/byte vs the H100 ridge point ~295):",
            "the SMs sit stalled waiting on HBM, so the GPU reads ~98% busy while doing ~20% of",
            "peak math. You pay for peak FLOPs and receive a fraction — the effective price per",
            "useful FLOP is 1/MFU times the list price:",
            "",
            "| GPU | Type | GPU-Util | MFU | Effective $/FLOP multiple |",
            "|---|---|---|---|---|",
        ]
        for lie in analysis.get("util_lies", []):
            lines.append(f"| {lie['gpu_id']} | {lie['gpu_type']} | {lie['gpu_util_pct']}% | "
                         f"{lie['mfu']:.3f} | {1.0 / lie['mfu']:.1f}x |")
        acts = analysis.get("actions", [])
        if acts:
            lines += [
                "",
                "## Recommended Actions (by ROI)",
                "",
                "| # | Action | Savings (USD/mo) | Effort | Why this order |",
                "|---|---|---|---|---|",
            ]
            for i, a in enumerate(acts, 1):
                lines.append(f"| {i} | {a['action']} | ${a['savings_usd']:,.0f} | "
                             f"{a['effort']} | {a['why']} |")
    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
    if reasoning:
        lines += [
            "",
            "## Reasoning Budget",
            "",
            f"- Reasoning traffic: {reasoning.get('requests_frac', 0):.1%} of requests, "
            f"{reasoning.get('tokens_frac', 0):.1%} of tokens",
            f"- Cost share: {reasoning.get('cost_frac', 0):.1%} of optimized spend; "
            f"energy share: {reasoning.get('wh_frac', 0):.1%} of Wh (~80x per query)",
            "",
            "| Cap | Rerouted | $ saved | Wh saved |",
            "|---|---|---|---|",
        ]
        for sc in reasoning.get("scenarios", []):
            lines.append(f"| {sc['cap']:.0%} | {sc['rerouted']} | ${sc['usd_saved']:,.2f} | {sc['wh_saved']:,.0f} |")
        lines += [
            "",
            "- Routing rule: budget-cap reasoning traffic; when over cap, reroute the **largest thinkers first** — keep reasoning only where it pays for itself.",
        ]
    if carbon:
        lines += [
            "",
            "## Carbon-Aware Scheduling",
            "",
            f"- Interruptible workload energy: {carbon.get('total_wh', 0) / 1000:,.0f} kWh/month",
            f"- Current ({carbon.get('current_region', 'n/a')}): {carbon.get('current_carbon_g', 0):,.0f} gCO2e/month",
            f"- Cleanest ({carbon.get('cleanest_region', 'n/a')}): {carbon.get('cleanest_carbon_g', 0):,.0f} gCO2e/month "
            f"-> save {carbon.get('saved_g', 0):,.0f} gCO2e ({carbon.get('saved_pct', 0):.0f}%)",
            "",
            "| Region | $/kWh | gCO2/kWh | Electricity | Carbon |",
            "|---|---|---|---|---|",
        ]
        for row in carbon.get("regions", []):
            lines.append(f"| {row['region']} | ${row['price_kwh']:.3f} | {row['gco2_kwh']} | "
                         f"${row['energy_cost_usd']:,.2f} | {row['carbon_g']:,.0f} |")
        regions = carbon.get("regions", [])
        cur = next((r for r in regions if r["region"] == carbon.get("current_region")), None)
        clean = next((r for r in regions if r["region"] == carbon.get("cleanest_region")), None)
        lines += [
            "",
            f"- Cheapest: {carbon.get('cheapest_region', 'n/a')} · Cleanest: {carbon.get('cleanest_region', 'n/a')} · "
            f"Balanced: {carbon.get('balanced_region', 'n/a')}",
        ]
        if cur and clean:
            delta = cur["energy_cost_usd"] - clean["energy_cost_usd"]
            verb = "also saves" if delta >= 0 else "costs an extra"
            lines.append(f"- Carbon meets cost: moving to {clean['region']} {verb} "
                         f"${abs(delta):,.0f}/month of electricity — the carbon cut is (nearly) free.")
        lines += [
            "- Latency trade-off: the cleanest region suits interruptible training/batch jobs;",
            "  latency-sensitive inference should stay near users (e.g. us-east-wa / us-west-2).",
        ]
    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a simple savings bar chart PNG. Returns the path. No-op if matplotlib absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals, color="#2e548a")
    ax.set_ylabel("Savings (USD / month)")
    ax.set_title("GPU cost savings by FinOps lever")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
