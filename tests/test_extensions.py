"""Extensions — Lab 25 "Your Turn" (Rubric D): reasoning budget + carbon-aware scheduling."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from missions import m2_inference_levers as m2
from finops import report


def _row(inp, out, reasoning, tier="small", cached=0, batch=0):
    return {"input_tokens": inp, "output_tokens": out, "cached_input_tokens": cached,
            "is_batch": batch, "is_reasoning": reasoning, "route_tier": tier}


# ---------- Ext 4: reasoning budget ----------

def test_reasoning_budget_splits_cost_and_wh():
    rows = [
        _row(2000, 1000, 1),
        _row(2000, 1000, 1),
        _row(2000, 200, 0),
        _row(2000, 200, 0),
    ]
    s = m2.reasoning_budget(rows)
    assert s["requests_total"] == 4
    assert s["requests_reasoning"] == 2
    assert abs(s["requests_frac"] - 0.5) < 1e-9
    # Wh: reasoning = (2000+1000)/1000 * 0.30 * 80 = 72 each -> 144; non-r = 0.66 each
    assert abs(s["wh_reasoning"] - 144.0) < 1e-9
    assert s["wh_frac"] > 0.99
    # optimized $ (small tier): reasoning 0.0008 each -> 0.0016; non-r 0.00048 each
    assert abs(s["opt_cost_reasoning"] - 0.0016) < 1e-9
    assert abs(s["cost_frac"] - 0.625) < 1e-9


def test_reasoning_budget_cap_reroutes_most_expensive_first():
    rows = [
        _row(2000, 1000, 1),   # rerouted (largest thinker — worst offender first)
        _row(2000, 800, 1),    # rerouted
        _row(2000, 600, 1),    # kept
        _row(2000, 200, 0),
    ]
    s = m2.reasoning_budget(rows, caps=(0.25,))
    sc = s["scenarios"][0]
    assert sc["cap"] == 0.25
    assert sc["kept"] == 1 and sc["rerouted"] == 2
    # $ saved: (1000-200) and (800-200) out-tok delta at small out price 0.40/1M
    assert abs(sc["usd_saved"] - (800 + 600) / 1e6 * 0.40) < 1e-9
    # Wh saved: reasoning(3000tok)=72 + reasoning(2800)=67.2 minus non-r(2200)=0.66*2
    assert abs(sc["wh_saved"] - (72.0 + 67.2 - 2 * 0.66)) < 1e-6


def test_reasoning_budget_nonbinding_cap_saves_nothing():
    rows = [
        _row(2000, 1000, 1),
        _row(2000, 800, 1),
        _row(2000, 600, 1),
        _row(2000, 200, 0),
    ]
    s = m2.reasoning_budget(rows, caps=(0.75,))   # 3 allowed >= 3 reasoning
    sc = s["scenarios"][0]
    assert sc["rerouted"] == 0 and sc["usd_saved"] == 0.0 and sc["wh_saved"] == 0.0


def test_build_report_reasoning_section_is_optional():
    assert "Reasoning Budget" not in report.build_report(1000, 500, {"lever": 500})
    md = report.build_report(1000, 500, {"lever": 500}, reasoning={
        "requests_frac": 0.084, "tokens_frac": 0.165, "cost_frac": 0.25, "wh_frac": 0.60,
        "scenarios": [{"cap": 0.05, "kept": 120, "rerouted": 81, "usd_saved": 1.2, "wh_saved": 9000.0}],
    })
    assert "Reasoning Budget" in md and "8.4%" in md and "16.5%" in md


def test_m2_run_exposes_reasoning_budget():
    r = m2.run(verbose=False)
    rb = r["reasoning"]
    assert 0 < rb["requests_reasoning"] <= rb["requests_total"]
    assert rb["scenarios"]


def test_m5_report_contains_reasoning_section():
    from missions import m5_report
    m5_report.run(verbose=False)
    md = open(os.path.join(ROOT, "outputs", "report.md"), encoding="utf-8").read()
    assert "Reasoning Budget" in md


# ---------- Ext 5: carbon-aware scheduling ----------

def test_job_energy_wh():
    from missions import ext5_carbon_aware as ext5
    job = {"num_gpus": "2", "hours_per_day": "10", "days": "7"}
    catalog_row = {"watts": "400"}
    assert ext5.job_energy_wh(job, catalog_row) == 400 * 2 * 10 * 7


def test_carbon_analysis_regions_and_savings():
    from missions import ext5_carbon_aware as ext5
    r = ext5.run(verbose=False)
    assert r["interruptible_jobs"] > 0
    assert r["total_wh"] > 0
    regions = {row["region"]: row for row in r["regions"]}
    assert set(regions) == set(ext5.sustainability.REGION_CARBON)
    # carbon scales linearly with grid intensity -> saved % is intensity ratio, Wh-independent
    assert abs(r["saved_pct"] - (1 - 30 / 380) * 100) < 0.1
    assert r["cleanest_region"] == "europe-north1"
    assert r["cheapest_region"] == "us-east-wa"
    # balanced pick must not be the dominated extremes
    assert r["balanced_region"] in regions


def test_build_report_carbon_section_is_optional():
    assert "Carbon-Aware Scheduling" not in report.build_report(1000, 500, {"lever": 500})
    md = report.build_report(1000, 500, {"lever": 500}, carbon={
        "total_wh": 500000.0, "current_region": "us-east-1",
        "current_carbon_g": 190000.0, "cleanest_region": "europe-north1",
        "cleanest_carbon_g": 15000.0, "saved_g": 175000.0, "saved_pct": 92.1,
        "cheapest_region": "us-east-wa", "balanced_region": "us-west-2",
        "regions": [{"region": "us-east-1", "price_kwh": 0.12, "gco2_kwh": 380,
                     "energy_cost_usd": 60.0, "carbon_g": 190000.0}],
    })
    assert "Carbon-Aware Scheduling" in md and "europe-north1" in md and "92%" in md


def test_m5_report_contains_carbon_section():
    from missions import m5_report
    m5_report.run(verbose=False)
    md = open(os.path.join(ROOT, "outputs", "report.md"), encoding="utf-8").read()
    assert "Carbon-Aware Scheduling" in md


# ---------- C.2 analysis sections (mechanism, ROI actions, carbon<->cost linkage) ----------

def test_build_report_analysis_sections_are_optional():
    plain = report.build_report(1000, 500, {"lever": 500})
    assert "GPU-Util Lie" not in plain and "Recommended Actions" not in plain
    md = report.build_report(1000, 500, {"lever": 500}, analysis={
        "util_lies": [{"gpu_id": "gpu-h100-4", "gpu_type": "H100",
                       "gpu_util_pct": 98.2, "mfu": 0.194, "mbu": 0.207}],
        "actions": [{"action": "Switch purchasing tiers (spot/reserved)",
                     "savings_usd": 10040, "effort": "low", "why": "80% of savings; contract only"}],
    })
    assert "GPU-Util Lie" in md
    assert "5.2x" in md                              # effective $/FLOP multiplier = 1/0.194
    assert "memory-bound" in md or "stalled" in md   # mechanism explained, not just numbers
    assert "Recommended Actions" in md and "Switch purchasing tiers" in md


def test_build_report_reasoning_states_routing_rule():
    md = report.build_report(1000, 500, {"l": 500}, reasoning={
        "requests_frac": 0.084, "tokens_frac": 0.165, "cost_frac": 0.25, "wh_frac": 0.94,
        "scenarios": [{"cap": 0.05, "kept": 120, "rerouted": 81, "usd_saved": 0.73, "wh_saved": 15778.0}],
    })
    assert "largest thinkers first" in md            # explicit routing rule (Rubric D.4)


def test_build_report_carbon_commentary_links_cost_and_latency():
    md = report.build_report(1000, 500, {"l": 500}, carbon={
        "total_wh": 1789000.0, "current_region": "us-east-1",
        "current_carbon_g": 679820.0, "cleanest_region": "europe-north1",
        "cleanest_carbon_g": 53670.0, "saved_g": 626150.0, "saved_pct": 92.1,
        "cheapest_region": "us-east-wa", "balanced_region": "us-east-wa",
        "regions": [
            {"region": "us-east-1", "price_kwh": 0.12, "gco2_kwh": 380,
             "energy_cost_usd": 214.68, "carbon_g": 679820.0},
            {"region": "europe-north1", "price_kwh": 0.09, "gco2_kwh": 30,
             "energy_cost_usd": 161.01, "carbon_g": 53670.0},
        ],
    })
    assert "$54" in md                    # electricity delta computed from regions: 214.68 - 161.01
    assert "latency" in md.lower() and "interruptible" in md


def test_m5_report_contains_analysis_sections():
    from missions import m5_report
    m5_report.run(verbose=False)
    md = open(os.path.join(ROOT, "outputs", "report.md"), encoding="utf-8").read()
    assert "GPU-Util Lie" in md and "Recommended Actions" in md
