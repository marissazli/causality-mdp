# fig2_manual.py
"""Regenerate Figure 2 with FAW hardcoded to the reported 38-row subset.
TP/CG/MAD counts are pulled live from outputs/per_row.csv."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── FAW hardcoded (38-row paper subset; rest of CSV is out of order) ─────────
COUNTS = {
    ("financial_article_writing", False): (0, 4, 1, 33),
    ("financial_article_writing", True ): (0, 1, 0, 37),
}

LIVE_ENVS = ["travel_planning", "code_generation", "multi_agent_debate"]
ENV_ORDER = ["travel_planning", "code_generation",
             "multi_agent_debate", "financial_article_writing"]
ENV_LABELS = {
    "travel_planning":           "Travel Planning",
    "code_generation":           "Code Generation",
    "multi_agent_debate":        "Multi-Agent Debate",
    "financial_article_writing": "Financial Article Writing",
}

CATS = ["Harm preserved", "Harm prevented", "Harm induced", "No harm in either"]
CAT_COLORS = {
    "Harm preserved":    "#b2182b",
    "Harm prevented":    "#4393c3",
    "Harm induced":      "#f4a582",
    "No harm in either": "#dddddd",
}


def load_live_counts(per_row_path="outputs/per_row.csv"):
    per_row = pd.read_csv(per_row_path)
    base = per_row[per_row["cf_mode"] == "baseline"]
    for env in LIVE_ENVS:
        env_sub = base[base["environment"] == env]
        for safe in [False, True]:
            sub = env_sub[env_sub["safe"] == safe]
            if len(sub) == 0:
                continue
            pres = int(((sub.y_factual == 1) & (sub.y_cf_mean == 1)).sum())
            prev = int(((sub.y_factual == 1) & (sub.y_cf_mean == 0)).sum())
            ind  = int(((sub.y_factual == 0) & (sub.y_cf_mean == 1)).sum())
            none = int(((sub.y_factual == 0) & (sub.y_cf_mean == 0)).sum())
            COUNTS[(env, safe)] = (pres, prev, ind, none)
            print(f"  {env:30s} safe={safe!s:5s}  "
                  f"pres={pres} prev={prev} ind={ind} none={none}  n={pres+prev+ind+none}")


def make_fig2(output_dir="outputs/"):
    print("Loading live counts from per_row.csv...")
    load_live_counts()
    print(f"FAW hardcoded: {COUNTS[('financial_article_writing', False)]}, "
          f"{COUNTS[('financial_article_writing', True)]}")

    os.makedirs(output_dir, exist_ok=True)
    n_env = len(ENV_ORDER)
    fig, axes = plt.subplots(n_env, 1, figsize=(8.0, 1.2 * n_env + 0.6),
                             sharex=False, squeeze=False)
    axes = axes[:, 0]

    for ax, env in zip(axes, ENV_ORDER):
        rows_data, labels = [], []
        for safe_val, lbl in [(False, "Non-safe"), (True, "Safe")]:
            counts = COUNTS.get((env, safe_val))
            if counts is None:
                continue
            pres, prev, ind, none = counts
            n = pres + prev + ind + none
            rows_data.append(dict(zip(CATS, [pres, prev, ind, none])))
            labels.append(f"{lbl}\n(n={n})")

        y_pos = np.arange(len(rows_data))
        left = np.zeros(len(rows_data))
        for c in CATS:
            vals = np.array([d[c] for d in rows_data])
            ax.barh(y_pos, vals, left=left, color=CAT_COLORS[c],
                    edgecolor="white", linewidth=0.8, label=c, height=0.6)
            for yi, v, l in zip(y_pos, vals, left):
                if v >= 2:
                    ax.text(l + v / 2, yi, str(v), ha="center", va="center",
                            fontsize=8.5,
                            color="white" if c != "No harm in either" else "#333")
            left += vals

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Scenarios" if env == ENV_ORDER[-1] else "")
        ax.set_title(ENV_LABELS[env], loc="left", fontsize=10, pad=2)
        ax.invert_yaxis()
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=8)

    handles = [mpatches.Patch(facecolor=CAT_COLORS[c], edgecolor="white", label=c)
               for c in CATS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=4, frameon=False, fontsize=9,
               handlelength=1.2, columnspacing=1.6)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pdf = os.path.join(output_dir, "fig2_transitions.pdf")
    png = os.path.join(output_dir, "fig2_transitions.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=150)
    print(f"saved {pdf}\nsaved {png}")


if __name__ == "__main__":
    make_fig2()