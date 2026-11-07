"""
make_figures.py

Generates paper figures from CSV outputs of evaluate_result.py.

Three figures (no redundancy with Tables 1-3 in the paper):

  fig1_framework.pdf       — Gumbel-Max tape schematic (no data, methods figure)
  fig2_transitions.pdf     — outcome (y_factual, y_cf) decomposition per env x safe
  fig3_harm_category.pdf   — causal effect by BAD-ACTS harm sub-category

Usage:
    python make_figures.py --csv-dir outputs/outputs_qwen/ --output-dir figures/figures_qwen/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import warnings
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
import paper_style as ps
ps.apply()

ENV_LABELS = {
    "travel_planning": "Travel Planning",
    "financial_article_writing": "Financial Article\nWriting",
    "code_generation": "Code Generation",
    "multi_agent_debate": "Multi-Agent Debate",
}
ENV_ORDER = ["travel_planning", "code_generation", "multi_agent_debate", "financial_article_writing"]


def save(fig, name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    png_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    print(f"  saved {pdf_path}")
    return pdf_path


# ── Figure 1: Framework schematic (Gumbel-Max tape) ──────────────────────────
def fig_framework(output_dir):
    """
    Schematic of the Gumbel-Max tape mechanism.

    Two parallel agent-call timelines (Factual, Counterfactual). Each call is a
    box; below each call sits a 'tape' strip whose cells are the per-token RNG
    states. In the CF run, the intervened call's tape span is skipped (greyed
    out) and its message is replaced; downstream calls reuse the same tape
    cells as the factual run.
    """
    fig, ax = plt.subplots(figsize=(14.8, 5.0))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    # ---- colours ----
    C_AGENT = "#cfe2f3"   # generic agent box
    C_AGENT_E = "#1f4e79"
    C_TARGET = "#fde0d2"  # target (harmful tool execution)
    C_TARGET_E = "#9c2f2f"
    C_INTERV = "#fff2cc"  # intervened call
    C_INTERV_E = "#bf9000"
    C_TAPE = "#dddddd"
    C_TAPE_USED = "#7f7f7f"
    C_TAPE_SKIP = "#ffd9d9"

    AGENT_LABELS = [r"$a_1$", r"$a_2$", "$a_k$\n(CF agent)", r"$a_{k+1}$", "$a_T$\n(target)"]
    AGENT_COLORS_F = [C_AGENT, C_AGENT, C_AGENT, C_AGENT, C_TARGET]
    AGENT_EDGES_F  = [C_AGENT_E, C_AGENT_E, C_AGENT_E, C_AGENT_E, C_TARGET_E]

    AGENT_COLORS_CF = [C_AGENT, C_AGENT, C_INTERV, C_AGENT, C_TARGET]
    AGENT_EDGES_CF  = [C_AGENT_E, C_AGENT_E, C_INTERV_E, C_AGENT_E, C_TARGET_E]

    # column x-centers and box dimensions (column spacing widened so inter-box arrows are visible)
    xs = [2.4, 4.7, 7.0, 9.3, 11.6]
    box_w, box_h = 1.85, 1.0

    def draw_call(ax, x, y, text, fc, ec, fontsize=16):
        b = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=3,
        )
        ax.add_patch(b)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                linespacing=1.0, zorder=4)

    def draw_tape(ax, x_center, y, n_cells=5, cell_color=C_TAPE_USED, edge="#444", label=None):
        cell_w = 0.20
        total_w = n_cells * cell_w
        x0 = x_center - total_w / 2
        for i in range(n_cells):
            r = Rectangle((x0 + i * cell_w, y - 0.15), cell_w * 0.92, 0.30,
                          facecolor=cell_color, edgecolor=edge, linewidth=0.6, zorder=3)
            ax.add_patch(r)
        if label:
            ax.text(x_center, y - 0.36, label, ha="center", va="top", fontsize=13, color="#333")

    # ---- Row labels (left margin, with breathing room before first agent box) ----
    ax.text(0.55, 4.3, "Factual\nrun", fontsize=16, fontweight="bold",
            color="#1f4e79", ha="center", va="center")
    ax.text(0.55, 1.7, "Counter-\nfactual\nrun", fontsize=16, fontweight="bold",
            color="#7c5b00", ha="center", va="center")

    # ---- Factual row ----
    y_factual = 4.3
    y_tape_f = 3.5
    for i, x in enumerate(xs):
        draw_call(ax, x, y_factual, AGENT_LABELS[i], AGENT_COLORS_F[i], AGENT_EDGES_F[i])
        draw_tape(ax, x, y_tape_f, n_cells=5, cell_color=C_TAPE_USED, edge="#333")
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - box_w / 2 - 0.02, y_factual),
                        xytext=(x + box_w / 2 + 0.02, y_factual),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=0.9), zorder=2)

    # outcome (placed right of the wider target column)
    ax.text(14.2, y_factual, r"$\Rightarrow\;y_{\mathrm{factual}}$",
            fontsize=16, va="center", ha="right")
    # tape label
    tape_label_x = xs[0] - box_w / 2 - 0.4
    ax.text(tape_label_x, y_tape_f, r"Tape $\mathcal{U}$:",
            fontsize=15, va="center", ha="right", style="italic", color="#333")

    # ---- Counterfactual row ----
    y_cf = 1.7
    y_tape_cf = 0.9
    tape_colors_cf = [C_TAPE_USED, C_TAPE_USED, C_TAPE_SKIP, C_TAPE_USED, C_TAPE_USED]
    tape_labels_cf = ["replay", "replay", "skip", "replay", "replay"]
    for i, x in enumerate(xs):
        draw_call(ax, x, y_cf, AGENT_LABELS[i], AGENT_COLORS_CF[i], AGENT_EDGES_CF[i])
        draw_tape(ax, x, y_tape_cf, n_cells=5,
                  cell_color=tape_colors_cf[i],
                  edge="#9c2f2f" if i == 2 else "#333",
                  label=tape_labels_cf[i])
        if i < len(xs) - 1:
            ax.annotate("", xy=(xs[i + 1] - box_w / 2 - 0.02, y_cf),
                        xytext=(x + box_w / 2 + 0.02, y_cf),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=0.9), zorder=2)
    ax.text(14.2, y_cf, r"$\Rightarrow\;y_{\mathrm{cf}}$",
            fontsize=16, va="center", ha="right")
    ax.text(tape_label_x, y_tape_cf, r"Tape $\mathcal{U}$:",
            fontsize=15, va="center", ha="right", style="italic", color="#333")

    # ---- Intervention callout above the intervened call ----
    intv_x = xs[2]
    ax.annotate(
        r"replace $m_{k^\ast}$ with neutral text",
        xy=(intv_x, y_cf + box_h / 2 + 0.02),
        xytext=(intv_x, y_cf + 1.15),
        ha="center", fontsize=14, color="#7c5b00",
        arrowprops=dict(arrowstyle="->", color="#7c5b00", lw=1.0, shrinkA=2, shrinkB=2),
    )

    # ---- CFE box on right ----
    cfe_x = 15.4
    cfe_y = (y_factual + y_cf) / 2
    box_left = cfe_x - 0.8             # = 14.6
    box_top = cfe_y + 0.65             # = 3.65
    box_bottom = cfe_y - 0.65          # = 2.35
    box = FancyBboxPatch(
        (box_left, box_bottom), 1.6, 1.3,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        facecolor="#f0f0f0", edgecolor="#333", linewidth=1.0, zorder=3,
    )
    ax.add_patch(box)
    ax.text(cfe_x, cfe_y + 0.27, r"$\mathrm{CFE}_{k^\ast}$",
            ha="center", va="center", fontsize=17)
    ax.text(cfe_x, cfe_y - 0.22, r"$=\;y_{\mathrm{cf}}{-}y_{\mathrm{factual}}$",
            ha="center", va="center", fontsize=14)

    # arrows: tail starts just right of the y label (which ends at x=14.2);
    # head sits just OUTSIDE the box's top-edge-center / bottom-edge-center, so the
    # arrows visibly diagonal in from above and below rather than appearing nearly
    # vertical along the left edge of the box.
    arrow_kw = dict(arrowstyle="->", color="#555", lw=0.9, shrinkA=2, shrinkB=0)
    ax.annotate("", xy=(cfe_x, box_top + 0.10), xytext=(14.35, y_factual),
                arrowprops=arrow_kw)
    ax.annotate("", xy=(cfe_x, box_bottom - 0.10), xytext=(14.35, y_cf),
                arrowprops=arrow_kw)

    # ---- Legend (bottom) ----
    legend_y = 0.25
    legend_items = [
        (mpatches.Patch(facecolor=C_AGENT, edgecolor=C_AGENT_E), "non-target agent call"),
        (mpatches.Patch(facecolor=C_TARGET, edgecolor=C_TARGET_E), "target agent (harmful tool)"),
        (mpatches.Patch(facecolor=C_INTERV, edgecolor=C_INTERV_E), "intervened call $k^\\ast$"),
        (mpatches.Patch(facecolor=C_TAPE_USED, edgecolor="#333"), "tape cells replayed"),
        (mpatches.Patch(facecolor=C_TAPE_SKIP, edgecolor="#9c2f2f"), "tape span skipped"),
    ]
    leg = ax.legend(
        handles=[h for h, _ in legend_items],
        labels=[l for _, l in legend_items],
        loc="lower center", bbox_to_anchor=(0.5, -0.05),
        ncol=5, frameon=False, handlelength=1.2, handleheight=1.0,
        columnspacing=1.4, borderpad=0.2, fontsize=13,
    )

    fig.tight_layout()
    return save(fig, "fig1_framework", output_dir)


# ── Figure 2: Outcome (y_factual, y_cf) transitions per env x safe ───────────
def fig_transitions(per_row, output_dir):
    """
    Outcome decomposition per environment and safety condition (neutral text
    intervention): each scenario's (y_factual, y_cf) pair is
      harm preserved (1,1), harm prevented (1,0), harm induced (0,1), no harm (0,0).
    Drawn at 0.85 of the text width, as included in the paper.
    """
    base = per_row[per_row["cf_mode"] == "baseline"].copy()
    if len(base) == 0:
        print("  [skip] no baseline rows for fig_transitions")
        return None
    base["y_factual"] = base["y_factual"].astype(int)

    def label_row(yf, ce):  # causal_effect_mean is in {-1, 0, +1} with one CF sample
        if yf == 1 and ce == 0:
            return "Harm preserved"
        if yf == 1 and ce <= -0.5:
            return "Harm prevented"
        if yf == 0 and ce >= 0.5:
            return "Harm induced"
        return "No harm in either run"

    base["category"] = [label_row(yf, ce) for yf, ce in zip(base["y_factual"], base["causal_effect_mean"])]
    cats = ["Harm preserved", "Harm prevented", "Harm induced", "No harm in either run"]
    colors = {"Harm preserved": ps.RED, "Harm prevented": ps.BLUE,
              "Harm induced": ps.GOLD, "No harm in either run": ps.GRAY}
    text_on = {"Harm preserved": "white", "Harm prevented": "white",
               "Harm induced": ps.INK, "No harm in either run": ps.INK}

    envs = [e for e in ENV_ORDER if e in base["environment"].values]
    fig, axes = plt.subplots(len(envs), 1, figsize=(0.85 * ps.TEXTWIDTH, 0.72 * len(envs) + 0.45),
                             squeeze=False, layout="constrained")
    for ax, env in zip(axes[:, 0], envs):
        sub = base[base["environment"] == env]
        rows, labels = [], []
        for safe_val, lbl in [(False, "Non-safe"), (True, "Safe")]:
            chunk = sub[sub["safe"] == safe_val]
            if len(chunk):
                rows.append({c: int((chunk["category"] == c).sum()) for c in cats})
                labels.append(lbl)
        n = sum(rows[0].values())
        y = np.arange(len(rows))
        left = np.zeros(len(rows))
        for c in cats:
            vals = np.array([r[c] for r in rows])
            ax.barh(y, vals, left=left, color=colors[c], edgecolor="white", linewidth=0.6,
                    height=0.62, label=c)
            for yi, v, l in zip(y, vals, left):
                if v >= 2:
                    ax.text(l + v / 2, yi, str(v), ha="center", va="center", fontsize=7,
                            color=text_on[c])
            left += vals
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, n)
        ax.set_title(f"{ENV_LABELS.get(env, env).replace(chr(10), ' ')} ($n={n}$)", loc="left")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=6))
    axes[-1, 0].set_xlabel("Scenarios")
    handles = [mpatches.Patch(facecolor=colors[c], edgecolor="none", label=c) for c in cats]
    fig.legend(handles=handles, loc="outside upper center", ncol=4)
    return ps.save(fig, os.path.join(output_dir, "fig2_transitions"))


# ── Figure 3: Harm sub-category causal effect (cleaned) ──────────────────────
def fig_harm(harm, output_dir):
    """
    Mean causal effect by BAD-ACTS harm subcategory, neutral text intervention,
    non-safe condition. One panel per environment with a nonzero effect, stacked
    vertically at full text width; subcategories with zero effect are omitted.
    Negative (harm prevented) is blue and positive (harm induced) red, as in fig2.
    """
    base = harm[(harm["cf_mode"] == "baseline") & (harm["safe"] == False) &
                harm["Sub-Category"].notna()].copy()
    if len(base) == 0:
        print("  [skip] no harm category data")
        return None
    panels = []
    for env in ENV_ORDER:
        sub = base[base["environment"] == env]
        cat = (sub.groupby("Sub-Category")
                  .agg(ase=("causal_effect_mean", "mean"), n=("causal_effect_mean", "count"))
                  .reset_index())
        cat = cat[cat["ase"].abs() > 0].sort_values("ase")
        if len(cat):
            panels.append((env, cat))
    if not panels:
        print("  [skip] all envs have zero effect across categories")
        return None

    lim = max(c["ase"].abs().max() for _, c in panels) * 1.08
    heights = [len(c) + 1.2 for _, c in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(ps.TEXTWIDTH, 0.17 * sum(heights) + 0.35),
                             gridspec_kw={"height_ratios": heights}, sharex=True,
                             squeeze=False, layout="constrained")
    for ax, (env, cat) in zip(axes[:, 0], panels):
        y = np.arange(len(cat))
        colors = [ps.BLUE if v < 0 else ps.RED for v in cat["ase"]]
        ax.barh(y, cat["ase"], color=colors, height=0.62, zorder=2)
        ax.set_yticks(y, [f"{c[:1].upper() + c[1:].replace('_', ' ')} ($n={k}$)" for c, k in zip(cat["Sub-Category"], cat["n"])])
        ax.axvline(0, color=ps.INK, linewidth=0.6, zorder=3)
        ax.grid(axis="x", color=ps.GRID, linewidth=0.5, zorder=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(cat) - 0.4)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(ENV_LABELS.get(env, env).replace("\n", " "), loc="left")
    axes[-1, 0].set_xlabel("ACE (negative: harm prevented, positive: harm induced)")
    return ps.save(fig, os.path.join(output_dir, "fig3_harm_category"))


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="outputs/outputs_qwen/")
    parser.add_argument("--output-dir", default="figures/figures_qwen/")
    parser.add_argument("--skip-framework", action="store_true",
                        help="skip the no-data framework schematic")
    args = parser.parse_args()

    if not args.skip_framework:
        print("Generating framework figure (no data needed)...")
        fig_framework(args.output_dir)

    print("\nLoading CSVs...")
    per_row_path = os.path.join(args.csv_dir, "per_row.csv")
    harm_path = os.path.join(args.csv_dir, "harm_category.csv")

    if os.path.exists(per_row_path):
        per_row = pd.read_csv(per_row_path)
        if "safe" in per_row.columns:
            per_row["safe"] = per_row["safe"].astype(str).str.lower().isin(["true", "1"])
        print("\nGenerating outcome-transitions figure...")
        fig_transitions(per_row, args.output_dir)
    else:
        print(f"  [skip] {per_row_path} not found — transitions figure not generated")

    if os.path.exists(harm_path):
        harm = pd.read_csv(harm_path)
        if "safe" in harm.columns:
            harm["safe"] = harm["safe"].astype(str).str.lower().isin(["true", "1"])
        print("\nGenerating harm-subcategory figure...")
        fig_harm(harm, args.output_dir)
    else:
        print(f"  [skip] {harm_path} not found — harm figure not generated")

    print(f"\nDone. Figures written to {args.output_dir}")