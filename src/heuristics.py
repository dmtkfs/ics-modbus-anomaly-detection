import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import os

# Create necessary directories
os.makedirs("data/baselines", exist_ok=True)
os.makedirs("figures/heuristics", exist_ok=True)


class ModbusHeuristicsDetector:
    def __init__(self):
        self.baseline_write_ratios = {}
        self.baseline_fc_distributions = {}
        self.write_codes = [5, 6, 15, 16]  # Write function codes

    def load_data(self, filepath="data/processed/master.csv"):
        """Load only the columns we actually use, with tight dtypes.
        Accept both 'AttackFamily' and 'Attack Family' and normalize to 'AttackFamily'.
        """
        # Canonical names we want the rest of the code to use
        canon_usecols = [
            "Source",
            "Label",
            "AttackFamily",  # canonical
            "FunctionCodeNum",
            "Destination Port",
            "Source Port",
        ]
        dtypes_canon = {
            "Source": "category",
            "Label": "category",
            "AttackFamily": "category",
            "FunctionCodeNum": "int16",
            "Destination Port": "Int32",
            "Source Port": "Int32",
        }

        # Column aliases we accept when reading
        aliases = {
            "AttackFamily": ["AttackFamily", "Attack Family"],
        }

        # Sniff header to decide which names are present
        header = pd.read_csv(filepath, nrows=0)
        header_cols = set(header.columns)

        # Build the actual usecols for read_csv (union of canon & accepted aliases)
        read_usecols = set(c for c in canon_usecols if c in header_cols)
        for canon, opts in aliases.items():
            for opt in opts:
                if opt in header_cols:
                    read_usecols.add(opt)

        # Map dtypes to the names that will actually be read
        dtypes_read = {}
        for col in read_usecols:
            # if it's an alias of a canonical col, use the canonical dtype
            if col == "Attack Family":
                dtypes_read[col] = dtypes_canon["AttackFamily"]
            else:
                dtypes_read[col] = dtypes_canon.get(col, None)
        dtypes_read = {k: v for k, v in dtypes_read.items() if v is not None}

        # Read (pyarrow if available; fall back to pandas engine)
        try:
            df = pd.read_csv(
                filepath,
                usecols=list(read_usecols),
                dtype=dtypes_read,
                engine="pyarrow",
            )
        except Exception:
            df = pd.read_csv(
                filepath,
                usecols=list(read_usecols),
                dtype=dtypes_read,
                low_memory=False,
            )

        # Normalize aliases → canonical column names
        if "Attack Family" in df.columns and "AttackFamily" not in df.columns:
            df = df.rename(columns={"Attack Family": "AttackFamily"})

        # If still missing, create the column for schema parity
        if "AttackFamily" not in df.columns:
            df["AttackFamily"] = "Unknown"
            df["AttackFamily"] = df["AttackFamily"].astype("category")

        return df

    def compute_baselines(self, df):
        print("Step 2: Computing baselines...")
        benign_df = df[df["Label"] == "Benign"].copy()

        # H1 Baseline: Write ratios per source
        for source in benign_df["Source"].unique():
            source_data = benign_df[benign_df["Source"] == source]
            write_count = len(
                source_data[source_data["FunctionCodeNum"].isin(self.write_codes)]
            )
            total_count = len(source_data)
            write_ratio = write_count / total_count if total_count > 0 else 0
            self.baseline_write_ratios[source] = write_ratio

        # H2 Baseline: Function code distributions per source
        for source in benign_df["Source"].unique():
            source_data = benign_df[benign_df["Source"] == source]
            fc_counts = source_data["FunctionCodeNum"].value_counts()
            fc_freq = fc_counts / fc_counts.sum()
            self.baseline_fc_distributions[source] = fc_freq

        # Save baselines
        with open("data/baselines/baseline_write_ratios.pkl", "wb") as f:
            pickle.dump(self.baseline_write_ratios, f)
        with open("data/baselines/baseline_fc_distributions.pkl", "wb") as f:
            pickle.dump(self.baseline_fc_distributions, f)

        print(f"Computed baselines for {len(self.baseline_write_ratios)} sources")

    def detect_h1_write_spikes(self, df, threshold_multiplier=3.0):
        """H1 (vectorized): compute per-source write ratio once, then compare to thresholds."""
        print("Step 3: Applying H1 (Write-Rate Spike) [vectorized]...")

        # current per-source ratios on the WHOLE df
        per_src_total = df.groupby("Source", observed=False).size()
        per_src_write = df.groupby("Source", observed=False)["FunctionCodeNum"].apply(
            lambda s: s.isin(self.write_codes).sum()
        )
        cur_ratio_by_src = (per_src_write / per_src_total).fillna(0.0)

        # baselines & thresholds
        baseline_series = pd.Series(self.baseline_write_ratios)  # index = Source
        all_write_ratios = (
            baseline_series.values if len(baseline_series) else np.array([0.0])
        )
        global_mean = float(np.mean(all_write_ratios))
        global_std = float(np.std(all_write_ratios))
        global_threshold = global_mean + threshold_multiplier * global_std

        # per-source threshold = baseline + k*global_std (fallback to global if no baseline)
        per_src_threshold = (
            baseline_series + threshold_multiplier * global_std
        ).reindex(cur_ratio_by_src.index)
        per_src_threshold = per_src_threshold.astype("float64").fillna(global_threshold)

        # flag per source, then map back to rows
        per_src_flag = (cur_ratio_by_src > per_src_threshold).astype(np.uint8)
        h1_flags = df["Source"].map(per_src_flag).astype(np.uint8).tolist()
        return h1_flags

    def detect_h2_variant_f(self, df, k_threshold=2.0):
        """H2 Variant F (vectorized): frequency outlier per (Source, FC) + mixed role directionality."""
        print("Step 4: Applying H2 Variant F (Composite Hybrid) [vectorized]...")

        # ---------- Part A: frequency outlier per (Source, FunctionCodeNum) ----------
        # current freq per (Source, FC)
        counts = (
            df.groupby(["Source", "FunctionCodeNum"], observed=False)
            .size()
            .rename("cnt")
        )
        totals = df.groupby("Source", observed=False).size().rename("tot")
        cur_freq = (counts / totals).rename("cur_freq").reset_index()

        # baseline freq per (Source, FC) from saved dicts
        # self.baseline_fc_distributions[source] is a Series of freq by FC for that source
        base_rows = []
        for src, s in self.baseline_fc_distributions.items():
            tmp = s.rename("base_freq").reset_index()
            tmp.columns = ["FunctionCodeNum", "base_freq"]
            tmp.insert(0, "Source", src)
            base_rows.append(tmp)
        if base_rows:
            base_freq = pd.concat(base_rows, ignore_index=True)
        else:
            base_freq = pd.DataFrame(columns=["Source", "FunctionCodeNum", "base_freq"])

        # join current with baseline
        freq_join = cur_freq.merge(
            base_freq, on=["Source", "FunctionCodeNum"], how="left"
        )
        # unseen FC in baseline → anomaly
        freq_join["freq_anom"] = False
        # deviation where we have a baseline
        has_base = freq_join["base_freq"].notna() & (freq_join["base_freq"] > 0)
        dev = (
            freq_join.loc[has_base, "cur_freq"] - freq_join.loc[has_base, "base_freq"]
        ).abs() / freq_join.loc[has_base, "base_freq"]
        freq_join.loc[has_base & (dev > k_threshold), "freq_anom"] = True
        # no baseline for this (src,fc) → anomalous
        freq_join.loc[freq_join["base_freq"].isna(), "freq_anom"] = True

        # map (Source, FC) → freq_anom flag back to rows
        freq_key = (
            freq_join["Source"].astype(str)
            + "|"
            + freq_join["FunctionCodeNum"].astype(str)
        )
        freq_map = dict(zip(freq_key, freq_join["freq_anom"].astype(np.uint8)))
        row_key = df["Source"].astype(str) + "|" + df["FunctionCodeNum"].astype(str)
        freq_flag_per_row = row_key.map(freq_map).fillna(0).astype(np.uint8)

        # ---------- Part B: directionality mixed-role check per Source ----------
        # master actions: dest port 502; slave actions: src port 502
        is_master = df["Destination Port"] == 502
        is_slave = df["Source Port"] == 502
        master_ct = is_master.groupby(df["Source"], observed=False).sum().rename("m")
        slave_ct = is_slave.groupby(df["Source"], observed=False).sum().rename("s")
        dir_df = pd.concat([master_ct, slave_ct], axis=1).fillna(0)
        dir_df["mixed"] = 0
        both = (dir_df["m"] > 0) & (dir_df["s"] > 0)
        dir_df.loc[both, "mixed"] = (
            ((dir_df["m"] / (dir_df["m"] + dir_df["s"])).between(0.1, 0.9))
        ).astype(np.uint8)

        dir_flag_per_row = df["Source"].map(dir_df["mixed"]).fillna(0).astype(np.uint8)

        # ---------- Combine ----------
        h2_flags = ((freq_flag_per_row | dir_flag_per_row).astype(np.uint8)).tolist()
        return h2_flags

    def evaluate_heuristics(self, df, h1_flags, h2_flags):
        """Step 5: Evaluate individual and combined heuristic performance"""
        print("Step 5: Evaluating heuristics...")

        # Convert labels to binary
        true_labels = (df["Label"] != "Benign").astype(int)

        # Combined heuristics (logical OR)
        combined_flags = [1 if (h1 or h2) else 0 for h1, h2 in zip(h1_flags, h2_flags)]

        # Calculate metrics
        results = {}

        # H1 metrics
        h1_precision = precision_score(true_labels, h1_flags, zero_division=0)
        h1_recall = recall_score(true_labels, h1_flags, zero_division=0)
        h1_f1 = f1_score(true_labels, h1_flags, zero_division=0)
        results["H1"] = {"precision": h1_precision, "recall": h1_recall, "f1": h1_f1}

        # H2 metrics
        h2_precision = precision_score(true_labels, h2_flags, zero_division=0)
        h2_recall = recall_score(true_labels, h2_flags, zero_division=0)
        h2_f1 = f1_score(true_labels, h2_flags, zero_division=0)
        results["H2"] = {"precision": h2_precision, "recall": h2_recall, "f1": h2_f1}

        # Combined metrics
        combined_precision = precision_score(
            true_labels, combined_flags, zero_division=0
        )
        combined_recall = recall_score(true_labels, combined_flags, zero_division=0)
        combined_f1 = f1_score(true_labels, combined_flags, zero_division=0)
        results["Combined"] = {
            "precision": combined_precision,
            "recall": combined_recall,
            "f1": combined_f1,
        }

        # Print results
        for method, metrics in results.items():
            print(
                f"{method}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
            )

        return results

    def visualize_results(self, df, h1_flags, h2_flags, results):
        """Step 6: Create visualizations"""
        print("Step 6: Creating visualizations...")

        # Performance comparison bar chart
        methods = list(results.keys())
        precisions = [results[method]["precision"] for method in methods]
        recalls = [results[method]["recall"] for method in methods]
        f1s = [results[method]["f1"] for method in methods]

        x = np.arange(len(methods))
        width = 0.25

        _fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, precisions, width, label="Precision", alpha=0.8)
        ax.bar(x, recalls, width, label="Recall", alpha=0.8)
        ax.bar(x + width, f1s, width, label="F1-Score", alpha=0.8)

        ax.set_xlabel("Heuristic Method")
        ax.set_ylabel("Score")
        ax.set_title("Heuristics Performance Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig("figures/heuristics/performance_comparison.png", dpi=300)
        plt.close()

        # Confusion matrix for combined heuristics (pretty formatting, no sci-notation)

        true_labels = (df["Label"] != "Benign").astype(int)
        combined_flags = [1 if (h1 or h2) else 0 for h1, h2 in zip(h1_flags, h2_flags)]

        cm = confusion_matrix(true_labels, combined_flags)

        # Make a dedicated figure with constrained layout to avoid overlap with colorbar
        fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)

        # Format colorbar ticks as plain integers with thousands separators
        fmt_int = FuncFormatter(lambda x, pos: f"{int(x):,}")

        sns.heatmap(
            cm,
            ax=ax,
            annot=True,
            fmt="d",  # integer annotations in the cells
            cmap="Blues",
            cbar_kws={"format": fmt_int},  # colorbar tick formatting
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - Combined Heuristics")

        # Also ensure axes tick labels don't go scientific
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):d}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):d}"))

        # Access the colorbar and force integer formatting (in case seaborn creates it late)
        cbar = ax.collections[0].colorbar
        cbar.formatter = fmt_int
        cbar.update_ticks()

        fig.savefig("figures/heuristics/confusion_combined.png", dpi=300)
        plt.close(fig)

        # Per-attack family analysis (vectorized)
        if "AttackFamily" in df.columns:
            fam = df["AttackFamily"].astype("category")
            y_true = (df["Label"] != "Benign").astype(int).to_numpy()
            h1_arr = np.asarray(h1_flags, dtype=np.uint8)
            h2_arr = np.asarray(h2_flags, dtype=np.uint8)
            comb_arr = (h1_arr | h2_arr).astype(np.uint8)

            # counts per family
            fam_codes = fam.cat.codes.to_numpy()
            n_fam = int(fam.cat.categories.size)

            def per_family_recall(pred):
                # TP per family = sum(pred & y_true), P per family = sum(y_true)
                tp = np.bincount(fam_codes, weights=(pred & y_true), minlength=n_fam)
                p = np.bincount(fam_codes, weights=y_true, minlength=n_fam)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(p > 0, tp / p, 0.0)
                return r

            r_h1 = per_family_recall(h1_arr)
            r_h2 = per_family_recall(h2_arr)
            r_comb = per_family_recall(comb_arr)

            fam_names = fam.cat.categories.tolist()
            x = np.arange(n_fam)
            width = 0.25

            _fig, ax = plt.subplots(figsize=(max(12, n_fam * 0.7), 6))
            ax.bar(x - width, r_h1, width, label="H1", alpha=0.8)
            ax.bar(x, r_h2, width, label="H2", alpha=0.8)
            ax.bar(x + width, r_comb, width, label="Combined", alpha=0.8)

            ax.set_xlabel("Attack Family")
            ax.set_ylabel("Recall")
            ax.set_title("Recall by Attack Family")
            ax.set_xticks(x)
            ax.set_xticklabels(fam_names, rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("figures/heuristics/recall_by_attack_family.png", dpi=300)
            plt.close()

    def export_results(self, df, h1_flags, h2_flags):
        """Step 7: Export results for ML team integration"""
        print("Step 7: Exporting results...")

        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("results", exist_ok=True)

        # Row-level outputs for downstream ML / analysis
        results_df = df[["Source", "Label", "AttackFamily"]].copy()
        results_df["H1_flag"] = h1_flags
        results_df["H2_flag"] = h2_flags
        results_df["Combined_flag"] = [
            1 if (h1 or h2) else 0 for h1, h2 in zip(h1_flags, h2_flags)
        ]

        # Optional Parquet (fast + compact)
        try:
            results_df.to_parquet(
                "data/processed/heuristics_results.parquet", index=False
            )
        except Exception:
            pass  # parquet optional

        # Human-readable CSV (no gzip)
        results_df.to_csv("data/processed/heuristics_results.csv", index=False)

        # EIP metrics summary
        y_true = (df["Label"] != "Benign").astype(int)
        combined_flags = [1 if (h1 or h2) else 0 for h1, h2 in zip(h1_flags, h2_flags)]

        metrics_df = pd.DataFrame(
            {
                "model": ["H1_WriteSpike", "H2_VariantF", "H1_H2_Combined"],
                "precision": [
                    precision_score(y_true, h1_flags, zero_division=0),
                    precision_score(y_true, h2_flags, zero_division=0),
                    precision_score(y_true, combined_flags, zero_division=0),
                ],
                "recall": [
                    recall_score(y_true, h1_flags, zero_division=0),
                    recall_score(y_true, h2_flags, zero_division=0),
                    recall_score(y_true, combined_flags, zero_division=0),
                ],
                "f1": [
                    f1_score(y_true, h1_flags, zero_division=0),
                    f1_score(y_true, h2_flags, zero_division=0),
                    f1_score(y_true, combined_flags, zero_division=0),
                ],
                "roc_auc": [None, None, None],
                "pr_auc": [None, None, None],
                "avg_loao_recall": [None, None, None],
                "notes": ["heuristics", "heuristics", "heuristics"],
            }
        )
        metrics_df[["precision", "recall", "f1"]] = metrics_df[
            ["precision", "recall", "f1"]
        ].round(3)
        metrics_df.to_csv("results/heuristics_metrics.csv", index=False)

        print("Results exported successfully!")
        return results_df


# Main execution function
def run_complete_heuristics():
    """Execute the complete 7-step heuristics workflow"""
    detector = ModbusHeuristicsDetector()

    # Execute all steps
    df = detector.load_data("data/processed/master.csv")
    detector.compute_baselines(df)
    h1_flags = detector.detect_h1_write_spikes(df)
    h2_flags = detector.detect_h2_variant_f(df)
    results = detector.evaluate_heuristics(df, h1_flags, h2_flags)
    detector.visualize_results(df, h1_flags, h2_flags, results)
    final_results = detector.export_results(df, h1_flags, h2_flags)

    print("\n=== COMPLETE HEURISTICS WORKFLOW FINISHED ===")
    print("✓ H1 (Write-Rate Spike) implemented")
    print("✓ H2 (Function-Code Anomaly - Variant F) implemented")
    print("✓ Combined heuristics evaluated")
    print("✓ Results exported for ML team integration")
    print("✓ All files ready for EIP compliance")

    return final_results


# Run the complete system
if __name__ == "__main__":
    results = run_complete_heuristics()
