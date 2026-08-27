"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def reasoning_budget(rows: list[dict], caps: tuple = (0.10, 0.05, 0.02)) -> dict:
    """Ext 4 — split optimized $ and energy Wh by is_reasoning; model traffic-cap scenarios.

    A capped (rerouted) reasoning request keeps its input/tier/batch flag but
    drops the thinking tokens: output shrinks to the non-reasoning average and
    the ~80x reasoning energy multiplier disappears. Reroute order is greedy —
    largest thinkers first.
    """
    n_total = len(rows)
    reasoning_rows = [r for r in rows if int(num(r["is_reasoning"])) == 1]
    plain_rows = [r for r in rows if int(num(r["is_reasoning"])) == 0]
    n_r = len(reasoning_rows)

    tok_r = tok_all = 0
    cost_r = cost_all = 0.0
    wh_r = wh_all = 0.0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        is_reasoning = int(num(r["is_reasoning"])) == 1
        pin, pout = MODEL_PRICES[r["route_tier"]]
        cost = pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
        wh = sustainability.wh_per_query(inp + out, is_reasoning=is_reasoning)
        tok_all += inp + out
        cost_all += cost
        wh_all += wh
        if is_reasoning:
            tok_r += inp + out
            cost_r += cost
            wh_r += wh

    # non-reasoning average output = the "thinking-free" output level
    plain_out_avg = (
        sum(int(num(r["output_tokens"])) for r in plain_rows) / len(plain_rows) if plain_rows else 0.0
    )

    scenarios = []
    by_out = sorted(reasoning_rows, key=lambda r: int(num(r["output_tokens"])), reverse=True)
    for cap in caps:
        allowed = int(cap * n_total)
        rerouted = by_out[: max(0, n_r - allowed)]
        usd = wh = 0.0
        for r in rerouted:
            inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
            _, pout = MODEL_PRICES[r["route_tier"]]
            batch_mult = 0.5 if bool(int(num(r["is_batch"]))) else 1.0
            usd += max(0.0, out - plain_out_avg) / 1e6 * pout * batch_mult
            wh += (
                sustainability.wh_per_query(inp + out, is_reasoning=True)
                - sustainability.wh_per_query(inp + round(plain_out_avg), is_reasoning=False)
            )
        scenarios.append({"cap": cap, "kept": min(n_r, allowed), "rerouted": len(rerouted),
                          "usd_saved": usd, "wh_saved": wh})

    return {
        "requests_total": n_total,
        "requests_reasoning": n_r,
        "requests_frac": n_r / n_total if n_total else 0.0,
        "tokens_reasoning": tok_r,
        "tokens_frac": tok_r / tok_all if tok_all else 0.0,
        "opt_cost_reasoning": cost_r,
        "opt_cost_total": cost_all,
        "cost_frac": cost_r / cost_all if cost_all else 0.0,
        "wh_reasoning": wh_r,
        "wh_total": wh_all,
        "wh_frac": wh_r / wh_all if wh_all else 0.0,
        "scenarios": scenarios,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    rb = reasoning_budget(rows)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print(f"\n-- Ext 4: Reasoning Budget --")
        print(f"reasoning : {rb['requests_reasoning']}/{rb['requests_total']} requests "
              f"({rb['requests_frac']:.1%}), {rb['tokens_frac']:.1%} of tokens")
        print(f"  cost share {rb['cost_frac']:.1%} of optimized $   energy share {rb['wh_frac']:.1%} of Wh")
        for sc in rb["scenarios"]:
            binding = "binding" if sc["rerouted"] else "non-binding (cap above current share)"
            print(f"  cap @{sc['cap']:.0%}: reroute {sc['rerouted']:>3} -> "
                  f"save ${sc['usd_saved']:,.2f} + {sc['wh_saved']:,.0f} Wh  [{binding}]")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "reasoning": rb,
    }


if __name__ == "__main__":
    run()
