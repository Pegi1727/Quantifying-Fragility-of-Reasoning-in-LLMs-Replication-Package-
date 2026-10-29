#!/usr/bin/env python3
"""Reproduce reported Tables 3–5 and verify the 50% stability horizon.

Source: attached paper, Tables 3, 4, and 5. Values below are transcribed
verbatim (including reported rounding); no auxiliary CSV is used because its
schema/data do not match the paper's reported tables.
"""
from math import exp, log
from pathlib import Path
import csv
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
MODELS = ["Llama-2-7B", "Llama-2-13B", "Llama-2-70B"]

# Table 3: Mean RPE ± reported 95% confidence interval, generations g0–g7.
# Values are strings to preserve exactly the paper's printed precision.
T3 = {
    "Llama-2-7B":  ["0.95 ±0.01", "0.81 ±0.02", "0.67 ±0.02", "0.52 ±0.03", "0.41 ±0.03", "0.34 ±0.03", "0.29 ±0.03", "0.25 ±0.03"],
    "Llama-2-13B": ["0.96 ±0.01", "0.86 ±0.02", "0.76 ±0.02", "0.66 ±0.03", "0.57 ±0.03", "0.50 ±0.03", "0.45 ±0.03", "0.41 ±0.03"],
    "Llama-2-70B": ["0.98 ±0.01", "0.91 ±0.02", "0.86 ±0.02", "0.81 ±0.02", "0.76 ±0.03", "0.72 ±0.03", "0.68 ±0.03", "0.64 ±0.03"],
}

# Table 4: RPE at g0, fragility/decay coefficient lambda, and R-squared.
T4 = {
    "Llama-2-7B":  {"parameters": "7B",  "RPE0": 0.95, "lambda": 0.22, "R_squared": 0.97},
    "Llama-2-13B": {"parameters": "13B", "RPE0": 0.96, "lambda": 0.16, "R_squared": 0.98},
    "Llama-2-70B": {"parameters": "70B", "RPE0": 0.98, "lambda": 0.10, "R_squared": 0.99},
}

# Table 5: reported horizon (generations), as printed in the paper.
T5 = {"Llama-2-7B": 3.1, "Llama-2-13B": 4.3, "Llama-2-70B": 6.9}


def print_table(title, headers, rows):
    print(f"\n{title}")
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def main():
    print("Reproducible paper-results summary | بازتولید نتایج گزارش‌شده مقاله")
    print("Source: paper Tables 3, 4, 5 | منبع: جداول ۳، ۴ و ۵ مقاله")

    rows3 = [[f"g{i}"] + [T3[m][i] for m in MODELS] for i in range(8)]
    print_table("Table 3 — Mean RPE ± 95% CI", ["Generation", *MODELS], rows3)
    rows4 = [[m, T4[m]["parameters"], f'{T4[m]["RPE0"]:.2f}', f'{T4[m]["lambda"]:.2f}', f'{T4[m]["R_squared"]:.2f}'] for m in MODELS]
    print_table("Table 4 — Exponential-degradation parameters", ["Model", "Parameters", "RPE0", "lambda", "R²"], rows4)

    rows5 = []
    print("\nTable 5 — Reported vs calculated 50% loss stability horizon")
    print("g* = ln(2)/lambda; compare rounded value (1 decimal) with Table 5.")
    for m in MODELS:
        lam = T4[m]["lambda"]
        calc = log(2) / lam
        reported = T5[m]
        # Table 4 prints lambda to 2 decimals while Table 5 prints horizon
        # to 1 decimal. Check whether some unrounded lambda could round to
        # the printed Table 4 coefficient and produce the printed horizon.
        implied_lambda = log(2) / reported
        lambda_rounding_interval = (lam - 0.005, lam + 0.005)
        horizon_rounding_interval = (log(2) / (reported + 0.05), log(2) / (reported - 0.05))
        compatible = max(lambda_rounding_interval[0], horizon_rounding_interval[0]) < min(lambda_rounding_interval[1], horizon_rounding_interval[1])
        assert compatible, (m, lam, reported, implied_lambda)
        note = "rounding-compatible / سازگار با گردکردن" if compatible else "MISMATCH / ناسازگار"
        rows5.append([m, f"{lam:.2f}", f"{calc:.6f} ({calc:.1f})", f"{reported:.1f}", note])
        print(f"{m}: displayed lambda gives g*={calc:.6f} (nearest tenth {calc:.1f}); "
              f"Table 5 reports {reported:.1f}; implied unrounded lambda={implied_lambda:.6f} "
              f"(prints as {implied_lambda:.2f}).")
    print_table("50% loss horizon | افق افت ۵۰٪", ["Model", "lambda", "ln(2)/lambda", "Table 5", "Check / بررسی"], rows5)
    print("\nAll Table 4/5 pairs are compatible at their printed precision; direct calculations use rounded lambda.")
    print("مقادیر جداول ۴ و ۵ در دقتِ چاپ‌شده سازگارند؛ محاسبه مستقیم از λ گرد‌شده استفاده می‌کند.")

    # Write clean CSV reproductions.
    with (OUT / "paper_table3_rpe.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Generation", *MODELS]); w.writerows(rows3)
    with (OUT / "paper_table4_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Model", "Parameters", "RPE0", "lambda", "R_squared"])
        for r in rows4: w.writerow(r)
    with (OUT / "paper_table5_horizon_check.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Model", "lambda", "Calculated_ln2_over_lambda", "Table5_reported", "Check"]); w.writerows(rows5)

    # Plot exact Table 3 observations and the Table 4 reported exponential fit
    # RPE(g) = RPE0 * exp(-lambda*g); fit curve is not forced through observations.
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    colors = ["#2463a6", "#e67e22", "#27864b"]
    gs = list(range(8))
    for m, color in zip(MODELS, colors):
        y = [float(v.split("±")[0].strip()) for v in T3[m]]
        ax.plot(gs, y, "o", color=color, label=f"{m} reported RPE")
        fine = [i / 20 for i in range(141)]
        fit = [T4[m]["RPE0"] * exp(-T4[m]["lambda"] * g) for g in fine]
        ax.plot(fine, fit, "-", color=color, alpha=.85,
                label=f"{m} fit (lambda={T4[m]['lambda']:.2f})")
        ax.axvline(log(2) / T4[m]["lambda"], color=color, linestyle=":", alpha=.35)
    ax.axhline(.5, color="black", linestyle="--", linewidth=1, label="50% of reference")
    ax.set(title="Reported RPE and exponential decay fits (Tables 3–5)",
           xlabel="Recursive generation g", ylabel="RPE")
    ax.set_xticks(range(8)); ax.set_ylim(0, 1.05); ax.grid(alpha=.25)
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(OUT / "paper_rpe_exponential_decay.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote: {OUT / 'paper_table3_rpe.csv'}")
    print(f"Wrote: {OUT / 'paper_table4_parameters.csv'}")
    print(f"Wrote: {OUT / 'paper_table5_horizon_check.csv'}")
    print(f"Wrote: {OUT / 'paper_rpe_exponential_decay.png'}")


if __name__ == "__main__":
    main()
