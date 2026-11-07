"""
make_table3.py

Builds Table 3 (prefix sensitivity) from the per-scenario output of estimate_gamma.py:
  table3_gamma.tex            option (a): sum of gamma, fraction < 1
  table3_delta_fallback.tex   option (b): sum of delta, fraction < 1/2
  prefix_sensitivity_gamma.csv  both models concatenated
  rerun_report.md             unit tests, acceptance checks A1 to A4, comparison table

Usage: python make_table3.py [--dir outputs/prefix_sensitivity]
"""

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

TEMPLATE = r"""\begin{table}[t]
\centering
\caption{Estimated Gumbel-Max disagreement $\sum \gamma^{(t,\ell)}$ over the first downstream call after $k^*$ (Proposition~1). ``LCE = JCE'' counts scenarios in which $\beta$ has a single call before the target, so that both interventions coincide and the sum is $0$; ``$<1$'' is the fraction of scenarios with a sum below $1$. Since $\gamma^{(t,\ell)} \ge \delta^{(t,\ell)}$ at every position, these sums are at least the corresponding sums of total-variation distances. N denotes the non-safe and S the safe condition; Code Generation is identical under both conditions by construction. Two Qwen Travel Planning scenarios have no call after $k^*$ and are omitted.}
\label{tab:prefix-sensitivity}
\small
\begin{tabular}{llrrrrrrrr}
\toprule
 & & \multicolumn{4}{c}{Qwen3-8B} & \multicolumn{4}{c}{Llama-3.1-8B-Instruct} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10}
Environment & & $n$ & LCE = JCE & Mean & $<1$ & $n$ & LCE = JCE & Mean & $<1$ \\
\midrule
Travel Planning & N & 39 & 12 & {{QWEN_TP_N_MEAN}} & {{QWEN_TP_N_LT}} & 40 & 1 & {{LLAMA_TP_N_MEAN}} & {{LLAMA_TP_N_LT}} \\
 & S & 39 & 11 & {{QWEN_TP_S_MEAN}} & {{QWEN_TP_S_LT}} & 40 & 3 & {{LLAMA_TP_S_MEAN}} & {{LLAMA_TP_S_LT}} \\
Code Generation & N/S & 52 & 2 & {{QWEN_CG_NS_MEAN}} & {{QWEN_CG_NS_LT}} & 52 & 1 & {{LLAMA_CG_NS_MEAN}} & {{LLAMA_CG_NS_LT}} \\
Multi-Agent Debate & N & 18 & 0 & {{QWEN_MAD_N_MEAN}} & {{QWEN_MAD_N_LT}} & 18 & 0 & {{LLAMA_MAD_N_MEAN}} & {{LLAMA_MAD_N_LT}} \\
 & S & 18 & 0 & {{QWEN_MAD_S_MEAN}} & {{QWEN_MAD_S_LT}} & 18 & 0 & {{LLAMA_MAD_S_MEAN}} & {{LLAMA_MAD_S_LT}} \\
Financial Article Writing & N & 66 & 3 & {{QWEN_FAW_N_MEAN}} & {{QWEN_FAW_N_LT}} & 66 & 14 & {{LLAMA_FAW_N_MEAN}} & {{LLAMA_FAW_N_LT}} \\
 & S & 66 & 1 & {{QWEN_FAW_S_MEAN}} & {{QWEN_FAW_S_LT}} & 66 & 10 & {{LLAMA_FAW_S_MEAN}} & {{LLAMA_FAW_S_LT}} \\
\midrule
All & & 350 & 31 & {{QWEN_ALL_MEAN}} & {{QWEN_ALL_LT}} & 352 & 30 & {{LLAMA_ALL_MEAN}} & {{LLAMA_ALL_LT}} \\
\bottomrule
\end{tabular}
\end{table}
"""

OLD_CAPTION = TEMPLATE[TEMPLATE.index(r"\caption{") + len(r"\caption{"): TEMPLATE.index("}\n\\label")]
FALLBACK_CAPTION = (r"Estimated prefix sensitivity $\sum \delta^{(t,\ell)}$ over the first downstream call after $k^*$ "
                    r"(Proposition~1). ``LCE = JCE'' counts scenarios in which $\beta$ has a single call before the "
                    r"target, so that both interventions coincide and the sum is $0$; ``$<1/2$'' is the fraction of "
                    r"scenarios with a sum below $1/2$, which suffices for an informative bound since "
                    r"$\gamma^{(t,\ell)} \le 2\delta^{(t,\ell)}/(1+\delta^{(t,\ell)})$. N denotes the non-safe and S "
                    r"the safe condition; Code Generation is identical under both conditions by construction. Two "
                    r"Qwen Travel Planning scenarios have no call after $k^*$ and are omitted.")

EXPECTED = {  # (n, LCE=JCE) per cell
    "qwen": {("TP", "N"): (39, 12), ("TP", "S"): (39, 11), ("CG", "NS"): (52, 2), ("MAD", "N"): (18, 0),
             ("MAD", "S"): (18, 0), ("FAW", "N"): (66, 3), ("FAW", "S"): (66, 1), ("ALL", ""): (350, 31)},
    "llama": {("TP", "N"): (40, 1), ("TP", "S"): (40, 3), ("CG", "NS"): (52, 1), ("MAD", "N"): (18, 0),
              ("MAD", "S"): (18, 0), ("FAW", "N"): (66, 14), ("FAW", "S"): (66, 10), ("ALL", ""): (352, 30)},
}
A1_MEANS = {  # current Table 3 sum-delta means
    "qwen": {("TP", "N"): 0.48, ("TP", "S"): 0.51, ("CG", "NS"): 0.95, ("MAD", "N"): 0.14, ("MAD", "S"): 0.20,
             ("FAW", "N"): 0.98, ("FAW", "S"): 0.96},
    "llama": {("TP", "N"): 0.94, ("TP", "S"): 0.88, ("CG", "NS"): 0.96, ("MAD", "N"): 1.35, ("MAD", "S"): 0.96,
              ("FAW", "N"): 0.75, ("FAW", "S"): 0.80},
}
ENV_LONG = {"TP": "travel_planning", "CG": "code_generation", "MAD": "multi_agent_debate",
            "FAW": "financial_article_writing"}


def cell(df, model, env, cond):
    d = df[df.model == model]
    if env == "ALL":
        return pd.concat([d, d[d.env == "CG"]])          # CG counted for N and for S
    return d[(d.env == env) & (d.condition == cond)]


def fill(df, col, thresh):
    vals = {}
    for model in EXPECTED:
        for env, cond in EXPECTED[model]:
            c = cell(df, model, env, cond)
            key = f"{model.upper()}_{env}" + (f"_{cond}" if cond else "")
            vals[key + "_MEAN"] = f"{c[col].mean():.2f}"
            vals[key + "_LT"] = f"{(c[col] < thresh).mean():.2f}"
    return vals


def render(vals, fallback):
    t = TEMPLATE
    if fallback:
        t = t.replace(OLD_CAPTION, FALLBACK_CAPTION).replace(r"Mean & $<1$", r"Mean & $<1/2$")
    t = re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], t)
    assert "{{" not in t
    return t


def compiles(tex):
    doc = "\\documentclass{article}\n\\usepackage{booktabs}\n\\begin{document}\n" + tex + "\\end{document}\n"
    with tempfile.TemporaryDirectory() as d:
        Path(d, "t.tex").write_text(doc)
        try:
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
                               cwd=d, capture_output=True, text=True)
        except FileNotFoundError:
            return None                                   # no LaTeX installed here
        return r.returncode == 0 and Path(d, "t.pdf").exists()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/prefix_sensitivity")
    args = ap.parse_args()
    out = Path(args.dir)
    df = pd.concat([pd.read_csv(out / f"prefix_sensitivity_gamma_{m}.csv") for m in EXPECTED], ignore_index=True)
    df["condition"] = df["condition"].astype(str)
    df[["model", "env", "condition", "scenario_idx", "k_star", "lce_eq_jce", "n_positions", "disagreed",
        "sum_delta", "sum_gamma", "sum_2d_over_1pd", "prompt_match"]].to_csv(
        out / "prefix_sensitivity_gamma.csv", index=False)

    # consistency gate
    mism = []
    for model, cells in EXPECTED.items():
        for (env, cond), (n, eq) in cells.items():
            c = cell(df, model, env, cond)
            if (len(c), int(c.lce_eq_jce.sum())) != (n, eq):
                mism.append(f"{model} {env} {cond}: got n={len(c)}, LCE=JCE={int(c.lce_eq_jce.sum())}; "
                            f"expected {n}, {eq}")
    if mism:
        print("CONSISTENCY GATE FAILED, tables not written:\n  " + "\n  ".join(mism))
        raise SystemExit(1)

    tex_g = render(fill(df, "sum_gamma", 1.0), fallback=False)
    tex_d = render(fill(df, "sum_delta", 0.5), fallback=True)
    (out / "table3_gamma.tex").write_text(tex_g)
    (out / "table3_delta_fallback.tex").write_text(tex_d)
    comp_g, comp_d = compiles(tex_g), compiles(tex_d)

    # ------------------------------------------------------------ report
    L = ["# Prefix sensitivity rerun (Table 3)", ""]
    L += ["## Unit tests", ""]
    for line in (out / "unit_tests.txt").read_text().splitlines():
        ok, name, msg = line.split("\t")
        L.append(f"- **{ok}** {name}: {msg}")

    # A1: compare with the original estimate_delta_v4 run
    a1_rows, n_diff_path, n_cmp = [], 0, 0
    for model in EXPECTED:
        old = pd.read_csv(f"outputs/outputs_{model}/delta_estimates.csv")
        old = old[old.delta_sum.notna()]
        for (env, cond), ref in A1_MEANS[model].items():
            c = cell(df, model, env, cond)
            a1_rows.append((model, env, cond, ref, c.sum_delta.mean()))
            o = old[(old.environment == ENV_LONG[env]) & (old.safe == (cond == "S"))]
            m = c.merge(o, left_on="scenario_idx", right_on="row_id")
            m = m[~m.lce_eq_jce_x]
            n_cmp += len(m)
            n_diff_path += int(((m.n_positions != m.n_steps) | (m.disagreed != (m.stop_reason == "diverged"))).sum())
    a1_ok = all(abs(new - ref) < 0.005 + 1e-9 for *_, ref, new in a1_rows)
    viol = int(df.bound_violations.sum())
    eq = df[df.lce_eq_jce]
    a3_ok = bool(((eq.sum_gamma == 0) & (eq.sum_delta == 0)).all())
    n_match = int(df.prompt_match.sum()) + int(df[df.env == "CG"].prompt_match.sum())
    n_total = len(df) + int((df.env == "CG").sum())

    L += ["", "## Acceptance checks", ""]
    L.append(f"- **A1 (reproduction): {'PASS' if a1_ok else 'NOT REPRODUCED'}.** "
             f"{n_diff_path} of {n_cmp} non-identical-prompt scenarios have a coupled path that differs from the "
             "original run (different length or disagreement status). This is expected: the original "
             "estimate_delta_v4.py did not use shared-noise Gumbel-Max but sampled one path of a simulated maximal "
             "coupling (disagree with probability TV, otherwise draw from min(p, q) normalized). The per-position "
             "delta is computed identically, and the original also included the disagreement position, so the "
             "summation convention is unchanged; only the sampled path differs. Means, current table vs rerun:")
    L += ["", "| Model | Env | Cond | Table 3 Σδ | Rerun Σδ |", "|---|---|---|---|---|"]
    L += [f"| {m} | {e} | {c} | {r:.2f} | {n:.2f} |" for m, e, c, r, n in a1_rows]
    L.append("")
    L.append(f"- **A2 (lemma bounds): {'PASS' if viol == 0 else 'FAIL'}.** {viol} positions violate "
             f"δ − 1e-9 ≤ γ ≤ 2δ/(1+δ) + 1e-9 (out of {int(df.n_positions.sum())} positions).")
    L.append(f"- **A3 (identical prompts): {'PASS' if a3_ok else 'FAIL'}.** {len(eq)} rows with LCE = JCE "
             "(CG counted once), all with Σγ = Σδ = 0." if a3_ok else f"- **A3: FAIL**")
    L.append(f"- **A4 (reconstruction): {'PASS' if n_match == 702 else 'FAIL'}.** {n_match}/{n_total} prompt "
             "matches (CG counted twice).")
    L.append("- Consistency gate: PASS. Both tables compile with booktabs: " +
             ("not checked (pdflatex is not installed on this machine)." if comp_g is None else
              f"gamma {'yes' if comp_g else 'NO'}, fallback {'yes' if comp_d else 'NO'}."))

    L += ["", "## Comparison", "",
          "| Model | Env | Cond | mean Σδ | mean Σγ | ratio | frac Σγ<1 | frac Σδ<0.5 |",
          "|---|---|---|---|---|---|---|---|"]
    for model in EXPECTED:
        for env, cond in EXPECTED[model]:
            c = cell(df, model, env, cond)
            md, mg = c.sum_delta.mean(), c.sum_gamma.mean()
            L.append(f"| {model} | {env} | {cond or '-'} | {md:.3f} | {mg:.3f} | "
                     f"{(mg / md if md else float('nan')):.3f} | {(c.sum_gamma < 1).mean():.2f} | "
                     f"{(c.sum_delta < 0.5).mean():.2f} |")

    df["gap"] = df.sum_gamma - df.sum_delta
    L += ["", "## Largest gaps (Σγ − Σδ)", "",
          "| Model | Env | Cond | Scenario | n_positions | disagreed | Σδ | Σγ | gap |", "|---|---|---|---|---|---|---|---|---|"]
    for r in df.nlargest(5, "gap").itertuples():
        L.append(f"| {r.model} | {r.env} | {r.condition} | {r.scenario_idx} | {r.n_positions} | {r.disagreed} | "
                 f"{r.sum_delta:.3f} | {r.sum_gamma:.3f} | {r.gap:.3f} |")
    (out / "rerun_report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
