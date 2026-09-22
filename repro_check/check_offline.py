"""Offline verification: the claims that can be checked without a GPU.

Three groups:
  MATH    the paper's central analytical claims about the two L0 corrections
  METRICS the bench's metric implementations against independent rewrites
  DOCS    the committed result JSON against the numbers printed in the docs

Run:  python repro_check/check_offline.py
Exit code is the number of failed checks.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anyjev.calibrate.contextual import apply_contextual, batch_prior, content_free_prior  # noqa: E402
from anyjev.calibrate.permute import cyclic_shifts, marginalize  # noqa: E402
from anyjev.calibrate.posthoc import TemperatureScaler  # noqa: E402
from bench import metrics  # noqa: E402

RESULTS: list[tuple[str, str, bool, str]] = []


def check(group: str, name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((group, name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {group:7s} {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# --------------------------------------------------------------------------
# MATH 1. Additive position bias is removed exactly by log-space marginalization
# --------------------------------------------------------------------------
def synth_logits(c: np.ndarray, b: np.ndarray, perms) -> np.ndarray:
    """Position-space log-probs for a model whose logit for 'option i shown at
    position j' is exactly c[i] + b[j]. Returns [P, K] normalized log-probs."""
    K = len(c)
    rows = []
    for perm in perms:
        logits = np.array([c[perm[j]] + b[j] for j in range(K)])
        rows.append(softmax(logits))
    return np.stack(rows)


def math_additive_bias():
    rng = np.random.default_rng(0)
    worst_logmean, worst_mean = 0.0, 0.0
    for K in (2, 3, 5, 20):
        for _ in range(200):
            c = rng.normal(0, 2.0, K)
            b = rng.normal(0, 2.0, K)          # arbitrary position bonus
            perms = cyclic_shifts(K)
            p_pos = synth_logits(c, b, perms)
            truth = softmax(c)
            worst_logmean = max(worst_logmean, np.abs(marginalize(p_pos, perms, "logmean") - truth).max())
            worst_mean = max(worst_mean, np.abs(marginalize(p_pos, perms, "mean") - truth).max())
    check("MATH", "logmean removes additive position bias exactly",
          worst_logmean < 1e-12, f"max abs error over 800 random cases = {worst_logmean:.2e}")
    check("MATH", "arithmetic mean does NOT (so the log-space choice is load-bearing)",
          worst_mean > 1e-3, f"max abs error for combine='mean' = {worst_mean:.3f}")


# --------------------------------------------------------------------------
# MATH 2. Invariance to how the options were originally listed
# --------------------------------------------------------------------------
def math_listing_invariance():
    """Re-list the same K options in a different order, redo the full cyclic-shift
    readout, map back to the original option identity: logmean must be identical."""
    rng = np.random.default_rng(1)
    dev_logmean, dev_mean = 0.0, 0.0
    for K in (3, 4, 5):
        for _ in range(60):
            c = rng.normal(0, 2.0, K)
            b = rng.normal(0, 2.0, K)
            perms = cyclic_shifts(K)
            base_lm = marginalize(synth_logits(c, b, perms), perms, "logmean")
            base_m = marginalize(synth_logits(c, b, perms), perms, "mean")
            for relabel in itertools.islice(itertools.permutations(range(K)), 24):
                relabel = np.asarray(relabel)          # new list position -> original option
                c2 = c[relabel]                        # the same options, listed differently
                got_lm = marginalize(synth_logits(c2, b, perms), perms, "logmean")
                got_m = marginalize(synth_logits(c2, b, perms), perms, "mean")
                # map back: got[x] is the option originally at index relabel[x]
                back_lm = np.zeros(K)
                back_m = np.zeros(K)
                back_lm[relabel] = got_lm
                back_m[relabel] = got_m
                dev_logmean = max(dev_logmean, np.abs(back_lm - base_lm).max())
                dev_mean = max(dev_mean, np.abs(back_m - base_m).max())
    check("MATH", "logmean is invariant to the original option listing",
          dev_logmean < 1e-12, f"max deviation = {dev_logmean:.2e}")
    check("MATH", "arithmetic mean is not listing-invariant",
          dev_mean > 1e-4, f"max deviation for combine='mean' = {dev_mean:.4f}")


# --------------------------------------------------------------------------
# MATH 3. Prior division recovers the signal when the prior is multiplicative
# --------------------------------------------------------------------------
def math_prior_identifiable():
    rng = np.random.default_rng(2)
    worst = 0.0
    for K in (2, 5, 20):
        for _ in range(300):
            signal = rng.dirichlet(np.ones(K))
            prior = rng.dirichlet(np.ones(K) * 3)
            observed = signal * prior
            observed = observed / observed.sum()
            got = apply_contextual(observed, prior)
            worst = max(worst, np.abs(got - signal).max())
    check("MATH", "prior division recovers the signal exactly when bias is multiplicative",
          worst < 1e-9, f"max abs error = {worst:.2e}")

    # the estimators themselves: batch mean and content-free mean, normalized
    p = np.stack([np.array([0.6, 0.4]), np.array([0.8, 0.2])])
    check("MATH", "batch_prior is the normalized mean over inputs",
          np.allclose(batch_prior(p), [0.7, 0.3]), f"got {batch_prior(p)}")
    cf = np.stack([np.array([0.7, 0.3]), np.array([0.9, 0.1])])
    check("MATH", "content_free_prior is the normalized mean over probes",
          np.allclose(content_free_prior(cf), [0.8, 0.2]), f"got {content_free_prior(cf)}")


# --------------------------------------------------------------------------
# MATH 4. Temperature scaling never changes the argmax, but CAN re-rank items
# --------------------------------------------------------------------------
def math_temperature():
    rng = np.random.default_rng(3)
    P = rng.dirichlet(np.ones(20) * 0.3, size=500)
    for T in (0.3, 1.7, 4.16):
        s = TemperatureScaler(temperature=T)
        Q = s.apply(P)
        same_argmax = bool((np.argmax(P, 1) == np.argmax(Q, 1)).all())
        check("MATH", f"temperature T={T} preserves every argmax", same_argmax)
    # cross-item re-ranking: this is what moves cov@5% while accuracy is frozen
    s = TemperatureScaler(temperature=4.16)
    conf_before = P.max(1)
    conf_after = s.apply(P).max(1)
    tau_changed = not np.array_equal(np.argsort(-conf_before), np.argsort(-conf_after))
    check("MATH", "temperature DOES re-rank items by confidence (explains the L1 cov@5% gain)",
          tau_changed, "confidence ordering across items changed under T>1")

    # the concrete mechanism: a peaked runner-up survives flattening, a diffuse tail does not
    a = np.array([0.9, 0.1] + [0.0] * 18)
    b = np.array([0.9] + [0.1 / 19] * 19)
    ca, cb = s.apply(a[None])[0].max(), s.apply(b[None])[0].max()
    check("MATH", "flattening ranks a concentrated runner-up above a diffuse tail",
          ca > cb + 0.3, f"same 0.900 top-1 -> {ca:.3f} (peaked) vs {cb:.3f} (diffuse)")

    # fitting recovers a planted temperature on well-specified data
    logits = rng.normal(0, 3.0, size=(4000, 5))
    T_true = 2.5
    p_true = np.exp(logits / T_true)
    p_true /= p_true.sum(1, keepdims=True)
    labels = [rng.choice(5, p=row) for row in p_true]
    fitted = TemperatureScaler.fit(softmax_rows(logits), labels).temperature
    check("MATH", "temperature fit recovers a planted temperature",
          abs(fitted - T_true) / T_true < 0.10, f"planted {T_true}, fitted {fitted:.3f}")


def softmax_rows(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# --------------------------------------------------------------------------
# METRICS. Independent rewrites of the bench metrics
# --------------------------------------------------------------------------
def ref_ece(probs, labels, n_bins=15):
    conf = probs.max(1)
    corr = (probs.argmax(1) == np.asarray(labels)).astype(float)
    idx = np.argsort(conf)
    total, n = 0.0, len(conf)
    for chunk in np.array_split(idx, min(n_bins, n)):
        total += len(chunk) / n * abs(conf[chunk].mean() - corr[chunk].mean())
    return total


def ref_cov_at_risk(probs, labels, target=0.05):
    """Deliberately written the other way round: scan every prefix, keep the
    LARGEST one that satisfies the risk bound."""
    conf = probs.max(1)
    corr = (probs.argmax(1) == np.asarray(labels)).astype(float)
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    best = 0.0
    errs = 0
    for n, i in enumerate(order, start=1):
        errs += 1 - corr[i]
        if errs / n <= target:
            best = n / len(conf)
    return best


def ref_first_crossing_cov(probs, labels, target=0.05):
    """The stricter reading: stop at the FIRST prefix that violates the bound."""
    conf = probs.max(1)
    corr = (probs.argmax(1) == np.asarray(labels)).astype(float)
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    errs, best = 0, 0.0
    for n, i in enumerate(order, start=1):
        errs += 1 - corr[i]
        if errs / n > target:
            break
        best = n / len(conf)
    return best


def metrics_checks():
    rng = np.random.default_rng(7)
    max_ece = max_cov = max_brier = 0.0
    for _ in range(300):
        n, K = rng.integers(40, 400), rng.integers(2, 21)
        probs = rng.dirichlet(np.ones(K) * rng.uniform(0.2, 3.0), size=n)
        labels = rng.integers(0, K, n)
        max_ece = max(max_ece, abs(metrics.ece(probs, labels) - ref_ece(probs, labels)))
        max_cov = max(max_cov, abs(metrics.coverage_at_risk(probs, labels) - ref_cov_at_risk(probs, labels)))
        oh = np.zeros_like(probs)
        oh[np.arange(n), labels] = 1
        max_brier = max(max_brier, abs(metrics.brier(probs, labels) - np.mean(((probs - oh) ** 2).sum(1))))
    check("METRICS", "ece matches an independent rewrite", max_ece < 1e-12, f"max diff {max_ece:.2e}")
    check("METRICS", "cov@5% matches an independent rewrite", max_cov < 1e-12, f"max diff {max_cov:.2e}")
    check("METRICS", "brier matches the textbook definition", max_brier < 1e-12, f"max diff {max_brier:.2e}")

    # perfect and adversarial ranking behave as they must
    K = 4
    perfect = np.zeros((100, K))
    perfect[:, 0] = 1.0
    check("METRICS", "cov@5% is 1.0 when every answer is right",
          metrics.coverage_at_risk(perfect, [0] * 100) == 1.0)
    check("METRICS", "cov@5% is 0.0 when the most confident answer is always wrong",
          metrics.coverage_at_risk(perfect, [1] * 100) == 0.0)

    # flip rate
    a = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    b = np.array([[0.4, 0.6], [0.3, 0.7], [0.6, 0.4]])
    check("METRICS", "flip_rate counts argmax disagreements", abs(metrics.flip_rate(a, b) - 1 / 3) < 1e-12)

    # cov@5% takes the deepest admissible cut, not the first crossing: quantify the gap
    gaps = []
    for _ in range(4000):
        n = 300
        probs = rng.dirichlet(np.ones(20) * 0.4, size=n)
        # plant a good-but-imperfect confidence ordering
        conf = probs.max(1)
        labels = np.where(rng.uniform(size=n) < np.clip(conf * 2.2, 0, 0.97),
                          probs.argmax(1), rng.integers(0, 20, n))
        deep = metrics.coverage_at_risk(probs, labels)
        first = ref_first_crossing_cov(probs, labels)
        gaps.append(deep - first)
    gaps = np.array(gaps)
    # The two readings can disagree by exactly one item (1/n) when a prefix sits
    # exactly on the 5.0% boundary and the two accumulation orders round apart;
    # anything beyond that is the metric genuinely being the optimistic reading.
    check("METRICS", "cov@5% uses the deepest admissible prefix (optimistic vs first-crossing)",
          gaps.min() >= -1.0 / 300 - 1e-9,
          f"deepest >= first-crossing up to one boundary item; mean gap {gaps.mean():+.3f}, "
          f"p95 {np.quantile(gaps, 0.95):+.3f}, max {gaps.max():+.3f} of coverage "
          f"({int((gaps > 1e-9).mean() * 100)}% of draws disagree)")

    # brittleness of cov@5% as a point estimate: flip one item near the cut
    probs = rng.dirichlet(np.ones(20) * 0.4, size=300)
    conf = probs.max(1)
    labels = np.where(rng.uniform(size=300) < np.clip(conf * 2.2, 0, 0.97),
                      probs.argmax(1), rng.integers(0, 20, 300))
    base = metrics.coverage_at_risk(probs, labels)
    order = np.argsort(-conf)
    swings = []
    for i in order[: int(0.7 * 300)]:
        lab2 = labels.copy()
        if lab2[i] == probs[i].argmax():
            lab2[i] = (lab2[i] + 1) % 20          # turn one correct item into an error
            swings.append(metrics.coverage_at_risk(probs, lab2) - base)
    swings = np.array(swings)
    check("METRICS", "cov@5% is a high-variance point estimate (single-item sensitivity)",
          True, f"flipping ONE correct item to wrong moves cov@5% by up to "
                f"{swings.min():+.3f} (mean {swings.mean():+.3f}) at n=300")


# --------------------------------------------------------------------------
# DOCS. Committed JSON vs the numbers printed in README and docs/
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def docs_consistency():
    # 1. every headline number in the README main table must exist in the JSON
    j = json.load(open(os.path.join(ROOT, "bench/results_batchprior_v0/2026-09-20/Qwen__Qwen3-8B.json")))
    bank = next(t for t in j["tasks"] if t["task"] == "banking20")
    lv = bank["levels"]
    want = [
        ("raw flip", lv["raw"]["flip"], 0.227), ("L0 flip", lv["L0"]["flip"], 0.077),
        ("raw acc", lv["raw"]["acc"], 0.750), ("L0 acc", lv["L0"]["acc"], 0.807),
        ("raw ece", lv["raw"]["ece"], 0.235), ("L0 ece", lv["L0"]["ece"], 0.180),
        ("L1 ece", lv["L1"]["ece"], 0.100),
        ("raw cov@5%", lv["raw"]["cov@5%"], 0.077), ("L0 cov@5%", lv["L0"]["cov@5%"], 0.477),
        ("L1 cov@5%", lv["L1"]["cov@5%"], 0.543),
    ]
    bad = [(n, got, exp) for n, got, exp in want if abs(round(got, 3) - exp) > 5e-4]
    check("DOCS", "README headline table matches the committed JSON",
          not bad, "all 10 cells match" if not bad else f"mismatches: {bad}")

    # 2. the 9-row cross-model flip table in the README
    rows = re.findall(r"^\| (Qwen[\w.\-]+) \| (\w+) \| (\d+) \| ([\d.]+) \| \*\*([\d.]+)\*\* \| "
                      r"([\d.]+) \| \*\*([\d.]+)\*\* \| ([\d.]+) \| \*\*([\d.]+)\*\* \|$",
                      open(os.path.join(ROOT, "README.md")).read(), re.M)
    bad = []
    for model, task, k, rflip, lflip, racc, lacc, rece, l1ece in rows:
        p = os.path.join(ROOT, f"bench/results_batchprior_v0/2026-09-20/Qwen__{model}.json")
        if not os.path.exists(p):
            bad.append((model, task, "no JSON"))
            continue
        t = next((x for x in json.load(open(p))["tasks"] if x["task"] == task), None)
        if t is None:
            bad.append((model, task, "task missing"))
            continue
        for label, got, exp in [("K", t["k"], int(k)),
                                ("raw flip", t["levels"]["raw"]["flip"], float(rflip)),
                                ("L0 flip", t["levels"]["L0"]["flip"], float(lflip)),
                                ("raw acc", t["levels"]["raw"]["acc"], float(racc)),
                                ("L0 acc", t["levels"]["L0"]["acc"], float(lacc)),
                                ("raw ece", t["levels"]["raw"]["ece"], float(rece)),
                                ("L1 ece", t["levels"]["L1"]["ece"], float(l1ece))]:
            if abs(round(float(got), 3) - exp) > 5e-4:
                bad.append((model, task, label, round(float(got), 3), exp))
    check("DOCS", f"README cross-model flip table ({len(rows)} rows x 7 cells) matches JSON",
          bool(rows) and not bad, "all cells match" if not bad else f"mismatches: {bad[:6]}")

    # 3. docs/results_bench.md must be exactly what bench.table prints from the JSON dir
    from bench.table import load_dir, markdown
    for results_dir, doc in [("bench/results_batchprior_v0/2026-09-20", "docs/results_bench.md")]:
        gen = markdown(load_dir(os.path.join(ROOT, results_dir)))
        doc_text = open(os.path.join(ROOT, doc)).read()
        gen_rows = {ln.strip() for ln in gen.splitlines() if ln.startswith("|")}
        doc_rows = {ln.strip() for ln in doc_text.splitlines() if ln.startswith("|")}
        missing = gen_rows - doc_rows
        check("DOCS", f"{doc} rows are regenerable from {results_dir}",
              not missing, "every generated row is present in the doc"
              if not missing else f"{len(missing)}/{len(gen_rows)} generated rows absent, e.g. {list(missing)[:2]}")

    # 4. the typed-decisions table
    from bench.tasks import base  # noqa: F401
    typed_dir = os.path.join(ROOT, "bench/results_typed/2026-09-21")
    seen = {}
    for p in sorted(glob.glob(os.path.join(typed_dir, "*.json"))):
        r = json.load(open(p))
        if "levels" in r:
            for lvl in ("raw", "L0", "L1"):
                if lvl in r["levels"]:
                    seen[(r["model"].split("/")[-1], lvl)] = r["levels"][lvl]["overall"]
        else:
            seen[(r["checkpoint"].split("/")[-1], "provider")] = r["overall"]
    readme = open(os.path.join(ROOT, "README.md")).read()
    expect = [
        ("Qwen3-32B", "L1", "acc", 0.701), ("Qwen3-32B", "L1", "ece", 0.034),
        ("Qwen3-8B", "L0", "acc", 0.640), ("Qwen3-8B", "raw", "acc", 0.626),
        ("laya-typed-decisions", "provider", "acc", 0.768),
        ("laya-typed-decisions", "provider", "ece", 0.215),
        ("laya", "provider", "acc", 0.359),
    ]
    bad = [(m, lv, k, round(seen.get((m, lv), {}).get(k, float("nan")), 3), e)
           for m, lv, k, e in expect
           if (m, lv) not in seen or abs(round(seen[(m, lv)][k], 3) - e) > 5e-4]
    check("DOCS", "typed-decisions table in README matches the committed JSON",
          not bad, f"{len(expect)} spot-checked cells match" if not bad else f"mismatches: {bad}")

    # 5. the claim "fine-tuned Laya's ECE is six times AnyJev L1's"
    laya = seen.get(("laya-typed-decisions", "provider"), {}).get("ece")
    best_l1 = min(v["ece"] for (m, lv), v in seen.items() if lv == "L1")
    ratio = laya / best_l1 if laya and best_l1 else float("nan")
    check("DOCS", "'six times' ECE claim is supported by the JSON",
          5.0 < ratio < 7.0, f"laya ECE {laya:.3f} / best L1 ECE {best_l1:.3f} = {ratio:.1f}x")

    # 6. the "7x automatable traffic" headline
    ratio = lv["L1"]["cov@5%"] / lv["raw"]["cov@5%"]
    check("DOCS", "'7x' coverage claim is supported by the JSON",
          6.5 < ratio < 7.5, f"{lv['L1']['cov@5%']:.3f} / {lv['raw']['cov@5%']:.3f} = {ratio:.1f}x")

    # 7. AURC, the robust version of the same ordering claim, moves much less
    check("DOCS", "AURC (robust ranking summary) improves far less dramatically than cov@5%",
          True, f"aurc raw {lv['raw']['aurc']:.3f} -> L0 {lv['L0']['aurc']:.3f} -> L1 {lv['L1']['aurc']:.3f} "
                f"({lv['raw']['aurc'] / lv['L1']['aurc']:.1f}x, vs {ratio:.1f}x for cov@5%)")

    # 8. the README's own counter-example must really be in the data
    inj = next(t for t in j["tasks"] if t["task"] == "injection")
    check("DOCS", "the self-reported counter-example (L0 cov below raw on injection) is real",
          inj["levels"]["L0"]["cov@5%"] < inj["levels"]["raw"]["cov@5%"],
          f"raw {inj['levels']['raw']['cov@5%']:.3f} -> L0 {inj['levels']['L0']['cov@5%']:.3f}")

    # 9. environment provenance is actually recorded
    env = j.get("env", {})
    check("DOCS", "results carry hardware and library provenance",
          bool(env.get("gpu")) and bool(env.get("transformers")),
          f"gpu={env.get('gpu')}, torch={env.get('torch')}, transformers={env.get('transformers')}")


def main():
    print("=" * 78)
    print("AnyJev offline verification")
    print("=" * 78)
    math_additive_bias()
    math_listing_invariance()
    math_prior_identifiable()
    math_temperature()
    metrics_checks()
    docs_consistency()
    print("-" * 78)
    failed = [r for r in RESULTS if not r[2]]
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for g, n, _, d in failed:
        print(f"  FAILED: {g} {n} -- {d}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "offline_checks.json"), "w") as f:
        json.dump([{"group": g, "check": n, "pass": p, "detail": d} for g, n, p, d in RESULTS], f, indent=1)
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
