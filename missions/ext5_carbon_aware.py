"""Ext 5 — Carbon-aware scheduling: move interruptible jobs to the best region.

Compares all regions on electricity price AND grid carbon, then quantifies the
gCO2e saved by relocating the interruptible (spot-friendly) workloads.

Run: python missions/ext5_carbon_aware.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import sustainability

CURRENT_REGION = "us-east-1"  # assumed deployment today


def job_energy_wh(job: dict, catalog_row: dict) -> float:
    """Energy of a workload job over its lifetime: watts x GPUs x hours x days."""
    return (num(catalog_row["watts"]) * num(job["num_gpus"])
            * num(job["hours_per_day"]) * num(job["days"]))


def _balanced_region(regions: list[dict]) -> str:
    """Region minimizing normalized (price + carbon) — a $/CO2 compromise."""
    max_p = max(r["price_kwh"] for r in regions) or 1.0
    max_c = max(r["gco2_kwh"] for r in regions) or 1.0
    return min(regions, key=lambda r: r["price_kwh"] / max_p + r["gco2_kwh"] / max_c)["region"]


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    inter = [j for j in jobs if int(num(j["interruptible"])) == 1]

    per_job = []
    total_wh = 0.0
    for j in inter:
        wh = job_energy_wh(j, cat[j["gpu_type"]])
        total_wh += wh
        per_job.append({"job_id": j["job_id"], "gpu_type": j["gpu_type"],
                        "wh": wh, "carbon_now_g": sustainability.carbon_g(wh, CURRENT_REGION)})

    regions = []
    for name, gco2 in sustainability.REGION_CARBON.items():
        regions.append({
            "region": name,
            "price_kwh": sustainability.REGION_PRICE_KWH[name],
            "gco2_kwh": gco2,
            "energy_cost_usd": sustainability.energy_cost_usd(total_wh, name),
            "carbon_g": sustainability.carbon_g(total_wh, name),
        })

    cleanest = min(regions, key=lambda r: r["gco2_kwh"])
    cheapest = min(regions, key=lambda r: r["price_kwh"])
    current_carbon = sustainability.carbon_g(total_wh, CURRENT_REGION)
    saved_g = current_carbon - cleanest["carbon_g"]
    saved_pct = saved_g / current_carbon * 100 if current_carbon else 0.0

    if verbose:
        print("== Ext 5: Carbon-Aware Scheduling ==")
        print(f"interruptible jobs: {len(inter)}  total energy {total_wh:,.0f} Wh ({total_wh / 1000:,.0f} kWh)")
        print(f"{'job':18}{'gpu':7}{'kWh':>9}{'gCO2e @' + CURRENT_REGION:>18}")
        for p in per_job:
            print(f"{p['job_id']:18}{p['gpu_type']:7}{p['wh'] / 1000:>9,.0f}{p['carbon_now_g']:>18,.0f}")
        print(f"\n{'region':16}{'$/kWh':>7}{'gCO2/kWh':>10}{'elec $':>9}{'gCO2e':>12}")
        for r in regions:
            print(f"{r['region']:16}{r['price_kwh']:>7.3f}{r['gco2_kwh']:>10}"
                  f"{r['energy_cost_usd']:>9.2f}{r['carbon_g']:>12,.0f}")
        print(f"\ncleanest: {cleanest['region']}  cheapest: {cheapest['region']}  "
              f"balanced: {_balanced_region(regions)}")
        print(f"move {CURRENT_REGION} -> {cleanest['region']}: save {saved_g:,.0f} gCO2e/month ({saved_pct:.1f}%)")

    return {
        "interruptible_jobs": len(inter),
        "total_wh": total_wh,
        "regions": regions,
        "current_region": CURRENT_REGION,
        "current_carbon_g": current_carbon,
        "cleanest_region": cleanest["region"],
        "cleanest_carbon_g": cleanest["carbon_g"],
        "cheapest_region": cheapest["region"],
        "balanced_region": _balanced_region(regions),
        "saved_g": saved_g,
        "saved_pct": saved_pct,
    }


if __name__ == "__main__":
    run()
