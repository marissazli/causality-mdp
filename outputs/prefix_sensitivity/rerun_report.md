# Prefix sensitivity rerun (Table 3)

## Unit tests

- **PASS** T1 brute force (V=50, 100 pairs): max abs err 6.66e-16
- **PASS** T2 known value: gamma 0.300000000000000, delta 0.250000000000000
- **PASS** T3 identity: gamma 0.0e+00, delta 0.0e+00
- **PASS** T4 Monte Carlo (V=1000, 1e6 draws): gamma 0.77504, empirical 0.77451, 3SE 0.00125

## Acceptance checks

- **A1 (reproduction): NOT REPRODUCED.** 249 of 540 non-identical-prompt scenarios have a coupled path that differs from the original run (different length or disagreement status). This is expected: the original estimate_delta_v4.py did not use shared-noise Gumbel-Max but sampled one path of a simulated maximal coupling (disagree with probability TV, otherwise draw from min(p, q) normalized). The per-position delta is computed identically, and the original also included the disagreement position, so the summation convention is unchanged; only the sampled path differs. Means, current table vs rerun:

| Model | Env | Cond | Table 3 Σδ | Rerun Σδ |
|---|---|---|---|---|
| qwen | TP | N | 0.48 | 0.46 |
| qwen | TP | S | 0.51 | 0.49 |
| qwen | CG | NS | 0.95 | 0.80 |
| qwen | MAD | N | 0.14 | 0.14 |
| qwen | MAD | S | 0.20 | 0.20 |
| qwen | FAW | N | 0.98 | 0.87 |
| qwen | FAW | S | 0.96 | 0.93 |
| llama | TP | N | 0.94 | 0.76 |
| llama | TP | S | 0.88 | 0.72 |
| llama | CG | NS | 0.96 | 0.93 |
| llama | MAD | N | 1.35 | 0.86 |
| llama | MAD | S | 0.96 | 0.88 |
| llama | FAW | N | 0.75 | 0.65 |
| llama | FAW | S | 0.80 | 0.68 |

- **A2 (lemma bounds): PASS.** 0 positions violate δ − 1e-9 ≤ γ ≤ 2δ/(1+δ) + 1e-9 (out of 6263 positions).
- **A3 (identical prompts): PASS.** 58 rows with LCE = JCE (CG counted once), all with Σγ = Σδ = 0.
- **A4 (reconstruction): PASS.** 702/702 prompt matches (CG counted twice).
- Consistency gate: PASS. Both tables compile with booktabs: not checked (pdflatex is not installed on this machine).

## Comparison

| Model | Env | Cond | mean Σδ | mean Σγ | ratio | frac Σγ<1 | frac Σδ<0.5 |
|---|---|---|---|---|---|---|---|
| qwen | TP | N | 0.460 | 0.465 | 1.011 | 0.87 | 0.54 |
| qwen | TP | S | 0.492 | 0.497 | 1.009 | 0.87 | 0.51 |
| qwen | CG | NS | 0.803 | 0.808 | 1.007 | 0.81 | 0.29 |
| qwen | MAD | N | 0.136 | 0.136 | 1.000 | 1.00 | 0.89 |
| qwen | MAD | S | 0.198 | 0.198 | 1.000 | 1.00 | 0.78 |
| qwen | FAW | N | 0.874 | 0.880 | 1.007 | 0.91 | 0.11 |
| qwen | FAW | S | 0.934 | 0.945 | 1.012 | 0.82 | 0.11 |
| qwen | ALL | - | 0.703 | 0.709 | 1.008 | 0.86 | 0.33 |
| llama | TP | N | 0.763 | 0.822 | 1.078 | 0.75 | 0.30 |
| llama | TP | S | 0.719 | 0.749 | 1.042 | 0.75 | 0.23 |
| llama | CG | NS | 0.930 | 0.984 | 1.058 | 0.65 | 0.25 |
| llama | MAD | N | 0.865 | 0.940 | 1.088 | 0.61 | 0.28 |
| llama | MAD | S | 0.875 | 0.957 | 1.094 | 0.50 | 0.22 |
| llama | FAW | N | 0.645 | 0.675 | 1.045 | 0.83 | 0.39 |
| llama | FAW | S | 0.679 | 0.704 | 1.038 | 0.80 | 0.35 |
| llama | ALL | - | 0.780 | 0.825 | 1.057 | 0.73 | 0.30 |

## Largest gaps (Σγ − Σδ)

| Model | Env | Cond | Scenario | n_positions | disagreed | Σδ | Σγ | gap |
|---|---|---|---|---|---|---|---|---|
| llama | TP | N | 17 | 28 | True | 3.098 | 3.557 | 0.460 |
| llama | CG | NS | 53 | 76 | True | 3.343 | 3.645 | 0.302 |
| llama | TP | N | 6 | 4 | True | 1.826 | 2.095 | 0.269 |
| llama | MAD | N | 5 | 45 | False | 2.162 | 2.424 | 0.262 |
| llama | MAD | N | 13 | 14 | True | 1.506 | 1.753 | 0.247 |
