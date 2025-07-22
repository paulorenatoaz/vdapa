#!/usr/bin/env python
import pandas as pd
from vdapa.config import config, BASE_DIR

# adjust path as needed
INPUT = BASE_DIR / config["paths"]["processed_data"] / "normalized_advisory.json"

# 1. load full dataset
df = pd.read_json(INPUT, lines=True)

# 2. compute these metrics before slicing to avoid SettingWithCopyWarning
df["num_cwes"] = df["cwe_ids"].apply(lambda x: len(x) if isinstance(x, list) else 0)
df["num_refs"] = df["references"].apply(lambda x: len(x) if isinstance(x, list) else 0)

# 3. null percentages
print("=== Null percentage per column ===")
print((df.isna().mean() * 100).sort_values(ascending=False), "\n")

# 4. subset with any time_to_review present (keep negatives!)
df_tr = df[df["time_to_review"].notna()].copy()

neg_count = (df_tr["time_to_review"] < 0).sum()
neg_pct   = (df_tr["time_to_review"] < 0).mean() * 100

print(f"Records with time_to_review: {len(df_tr)}  (of {len(df)})")
print(f"  → negatives: {neg_count} ({neg_pct:.1f}%)\n")

# 5. boolean distributions
for col in ["github_reviewed", "has_cve", "has_withdrawn"]:
    print(f"-- {col} --")
    vc = df_tr[col].value_counts(dropna=False)
    pct = vc / len(df_tr) * 100
    print(pd.DataFrame({"count": vc, "percent": pct}), "\n")

# 6. categorical distributions
for col in ["severity_label", "ecosystem"]:
    print(f"-- {col} --")
    vc = df_tr[col].value_counts(dropna=False)
    pct = vc / len(df_tr) * 100
    print(pd.DataFrame({"count": vc, "percent": pct}).head(10), "\n")

# 7. time_to_review summary
print("=== time_to_review summary ===")
print(df_tr["time_to_review"].describe(), "\n")



print("=== time_to_review percentiles ===")
for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
    q = df_tr["time_to_review"].quantile(p / 100)
    print(f" {p:>3}th pct: {q:.2f}")
print()

print("=== time_to_nvd summary ===")
print(df_tr["time_to_nvd"].describe(), "\n")
print("=== time_to_nvd percentiles ===")
for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
    q = df_tr["time_to_nvd"].quantile(p / 100)
    print(f" {p:>3}th pct: {q:.2f}")

# 8. num_cwes & num_refs summary
print("=== num_cwes & num_refs summary ===")
print(df_tr[["num_cwes", "num_refs"]].describe(), "\n")

# 9. top‐10 CWE families
cwe_pct = (
    df_tr
    .explode("cwe_ids")["cwe_ids"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .head(10)
)
print("=== Top‐10 CWE % ===")
print(cwe_pct, "\n")

# 10. correlations (include cyclical & cvss features if present)
features = ["time_to_review", "num_refs", "num_cwes", "year", "month_sin", "month_cos", "cvss_number"]
present = [c for c in features if c in df_tr.columns]
print("=== Correlation with time_to_review ===")
print(df_tr[present].corr()["time_to_review"].sort_values(ascending=False), "\n")

# 11. median time_to_review by groups
print("=== Median time_to_review by severity_label ===")
print(df_tr.groupby("severity_label")["time_to_review"].median(), "\n")

print("=== Median time_to_review by top ecosystems ===")
print(
    df_tr
    .groupby("ecosystem")["time_to_review"]
    .median()
    .sort_values(ascending=False)
    .head(10),
    "\n"
)

# 12. seasonality: median by month
if "month" in df_tr.columns:
    print("=== Median time_to_review by month ===")
    print(df_tr.groupby("month")["time_to_review"].median(), "\n")
