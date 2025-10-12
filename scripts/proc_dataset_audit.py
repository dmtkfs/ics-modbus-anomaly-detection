import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.dates import DateFormatter


# Paths
ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "data" / "processed" / "master.csv"
OUTDIR = ROOT / "figures" / "dataset_stats"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Helpers
fmt_int = FuncFormatter(lambda x, pos: f"{int(x):,}")


def bar_no_sci(ax):
    ax.yaxis.set_major_formatter(fmt_int)
    ax.xaxis.set_major_formatter(fmt_int)


def annotate_vertical_bars(ax):
    for p in ax.patches:
        v = int(p.get_height())
        ax.text(
            p.get_x() + p.get_width() / 2,
            p.get_height(),
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def annotate_horizontal_bars(ax):
    xmax = ax.get_xlim()[1]
    for p in ax.patches:
        v = int(p.get_width())
        ax.text(
            p.get_x() + p.get_width() + 0.005 * xmax,
            p.get_y() + p.get_height() / 2,
            f"{v:,}",
            ha="left",
            va="center",
            fontsize=9,
        )


def as_str(series):
    """Force categorical values to strings for plotting (prevents 0/1/2 axes)."""
    return series.astype(str)


# Load CSV
df = pd.read_csv(IN_CSV)

# Ensure names (not numeric codes) for labels/families
label_map = {0: "Attack", 1: "Benign", "0": "Attack", "1": "Benign"}
family_map = {
    0: "Compromised-scada",
    1: "Compromised-IED",
    2: "Benign",
    3: "External",
    "0": "Compromised-scada",
    "1": "Compromised-IED",
    "2": "Benign",
    "3": "External",
}
df["Label"] = df["Label"].map(lambda x: label_map.get(x, x))
df["Attack Family"] = df["Attack Family"].map(lambda x: family_map.get(x, x))

# Core stats
n_rows, n_cols = df.shape
cols = list(df.columns)
nulls = df.isna().sum()

lbl_counts = df["Label"].value_counts()
fam_counts = df["Attack Family"].value_counts()

fc_counts = df["Function Code"].value_counts()
fc_top8 = fc_counts.head(8)

tmin = pd.to_numeric(df["Time"]).min()
tmax = pd.to_numeric(df["Time"]).max()
duration_days = float(tmax - tmin) / (24 * 3600)

length_desc = df["Length"].describe()

uniq_src = df["Source"].nunique()
uniq_dst = df["Destination"].nunique()
top_src = df["Source"].value_counts().head(10)
top_dst = df["Destination"].value_counts().head(10)
top_pairs = (
    df.groupby(["Source", "Destination"]).size().sort_values(ascending=False).head(10)
)

# Query/response split by 502
is_query = df["Destination Port"] == 502
query_count = int(is_query.sum())
resp_count = int((~is_query).sum())

# Per-IP role split
ip_role = (
    pd.DataFrame(
        {
            "Queries_to_502": df[is_query].groupby("Source").size(),
            "Responses_from_502": df[~is_query].groupby("Destination").size(),
        }
    )
    .fillna(0)
    .astype(int)
    .sort_values("Queries_to_502", ascending=False)
)

# Per-day volume
ts = pd.to_datetime(df["Time"], unit="s")
daily = ts.dt.floor("D").value_counts().sort_index()

# Ports
top_sport = df["Source Port"].value_counts().head(15)
top_dport = df["Destination Port"].value_counts().head(15)

# Save CSV tables
(
    lbl_counts.rename_axis("Label")
    .to_frame("Count")
    .assign(Percent=(100 * lbl_counts / n_rows).round(2))
    .to_csv(OUTDIR / "labels.csv")
)

(
    fam_counts.rename_axis("Attack Family")
    .to_frame("Count")
    .assign(Percent=(100 * fam_counts / n_rows).round(2))
    .to_csv(OUTDIR / "attack_families.csv")
)

fc_counts.rename_axis("Function Code").to_frame("Count").to_csv(
    OUTDIR / "function_codes_all.csv"
)
fc_top8.rename_axis("Function Code").to_frame("Count").to_csv(
    OUTDIR / "function_codes_top8.csv"
)
top_src.rename_axis("Source").to_frame("Count").to_csv(OUTDIR / "top_sources.csv")
top_dst.rename_axis("Destination").to_frame("Count").to_csv(
    OUTDIR / "top_destinations.csv"
)
top_pairs.rename("Count").to_frame().to_csv(OUTDIR / "top_src_dst_pairs.csv")
ip_role.to_csv(OUTDIR / "per_ip_query_response.csv")
daily.rename("Count").to_frame().to_csv(OUTDIR / "per_day_counts.csv")
top_sport.rename_axis("Source Port").to_frame("Count").to_csv(
    OUTDIR / "top_source_ports.csv"
)
top_dport.rename_axis("Destination Port").to_frame("Count").to_csv(
    OUTDIR / "top_destination_ports.csv"
)

# Console output (full)
print(f"Reading: {IN_CSV}")
print(f"Rows: {n_rows:,} | Columns: {n_cols}")
print("\nColumns:", cols)
print("\nNulls per column (should be 0):")
print(nulls.to_string())

print("\nLabel counts:")
print(
    (
        lbl_counts.rename_axis("Label")
        .to_frame("Count")
        .assign(Percent=(100 * lbl_counts / n_rows).round(2))
    ).to_string()
)

print("\nAttack Family counts:")
print(
    (
        fam_counts.rename_axis("Attack Family")
        .to_frame("Count")
        .assign(Percent=(100 * fam_counts / n_rows).round(2))
    ).to_string()
)

print("\nTop Function Codes:")
print(fc_top8.to_frame("Count").to_string())

print(f"\nTime span: {duration_days:.2f} days (min={tmin}, max={tmax})")

print("\nLength summary:")
print(length_desc.to_string())

print(f"\nUnique IPs: sources={uniq_src:,}, destinations={uniq_dst:,}")
print("\nTop 10 Sources:\n" + top_src.to_string())
print("\nTop 10 Destinations:\n" + top_dst.to_string())
print("\nTop 10 Source→Destination pairs:\n" + top_pairs.to_string())

print(f"\nDirection (by port 502): queries={query_count:,}  responses={resp_count:,}")
print(f"\nSaved artifacts → {OUTDIR}")

# ---------- FIGURES ----------

# Mapping to canonical order with code prefixes in labels
label_order = ["Attack", "Benign"]
label_code = {"Attack": "0 - Attack", "Benign": "1 - Benign"}

lbl_named = lbl_counts.reindex(label_order).fillna(0).astype(int)
lbl_named.index = [label_code[i] for i in lbl_named.index]

# Label Bar
label_order = ["Attack", "Benign"]
label_code = {"Attack": "0 - Attack", "Benign": "1 - Benign"}

lbl_named = (
    df["Label"]
    .map(lambda x: {0: "Attack", 1: "Benign", "0": "Attack", "1": "Benign"}.get(x, x))
    .value_counts()
    .reindex(label_order)
    .fillna(0)
    .astype(int)
)
lbl_named.index = [label_code[i] for i in lbl_named.index]

plt.figure(figsize=(10, 5))
ax = plt.bar(range(len(lbl_named)), lbl_named.values)
plt.title("Label Distribution")
plt.ylabel("Count")
plt.xticks(range(len(lbl_named)), lbl_named.index, rotation=0)
# annotate numbers on bars
for i, v in enumerate(lbl_named.values):
    plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
# integer tick formatting
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTDIR / "labels.png", dpi=150)


# 2) Attack family distribution (named)
af_order = ["Compromised-scada", "Compromised-IED", "External", "Benign"]
af = fam_counts.reindex(af_order).dropna().astype(int)

# Attack family distribution bar
af_order = ["Compromised-scada", "Compromised-IED", "External", "Benign"]
af_counts = (
    df["Attack Family"]
    .map(
        lambda x: {
            0: "Compromised-scada",
            1: "Compromised-IED",
            2: "Benign",
            3: "External",
            "0": "Compromised-scada",
            "1": "Compromised-IED",
            "2": "Benign",
            "3": "External",
        }.get(x, x)
    )
    .value_counts()
)
af_counts = af_counts.reindex(af_order).dropna().astype(int)

plt.figure(figsize=(12, 5))
ax = plt.bar(range(len(af_counts)), af_counts.values)
plt.title("Attack Family Distribution")
plt.ylabel("Count")
plt.xticks(range(len(af_counts)), af_counts.index, rotation=0)
for i, v in enumerate(af_counts.values):
    plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
plt.tight_layout()
plt.savefig(OUTDIR / "attack_families.png", dpi=150)


# 2b) Attack family pie
plt.figure(figsize=(7, 7))
af.plot(
    kind="pie",
    autopct=lambda p: f"{p:.2f}%",
    ylabel="",
    title="Attack Family Distribution (Pie)",
)
plt.tight_layout()
plt.savefig(OUTDIR / "attack_families_pie.png", dpi=150)


# 3) Top function codes (horizontal + labels)
fc_name2num = {
    "Read Coils": 1,
    "Read Discrete Inputs": 2,
    "Read Holding Registers": 3,
    "Read Input Registers": 4,
    "Write Single Coil": 5,
    "Write Single Register": 6,
    "Write Multiple Coils": 15,
    "Write Multiple Registers": 16,
}

fc8_raw = df["Function Code"].value_counts().head(8)
# build labels with codes
fc8_labels = [f"{fc_name2num.get(name, '?')} - {name}" for name in fc8_raw.index]
# sort for barh (small to big) and reorder labels accordingly
order = fc8_raw.sort_values().index
fc8_vals_sorted = fc8_raw.loc[order].values
fc8_labels_sorted = [f"{fc_name2num.get(name, '?')} - {name}" for name in order]

plt.figure(figsize=(12, 7))
plt.barh(range(len(fc8_vals_sorted)), fc8_vals_sorted)
plt.title("Top 8 Function Codes")
plt.xlabel("Count")
plt.yticks(range(len(fc8_labels_sorted)), fc8_labels_sorted)
# annotate counts at bar ends
xmax = max(fc8_vals_sorted)
for i, v in enumerate(fc8_vals_sorted):
    plt.text(v + 0.005 * xmax, i, f"{v:,}", va="center", fontsize=9)

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x):,}"))
plt.tight_layout()
plt.subplots_adjust(left=0.35)
plt.savefig(OUTDIR / "function_codes_top8.png", dpi=150)


# 4) Packet length histogram (labeled)
plt.figure(figsize=(12, 7))
ax = df["Length"].plot(
    kind="hist", bins=80, title="Packet Length Histogram (log scale)"
)
ax.set_yscale("log")
ax.set_ylabel("Frequency (log scale)")
ax.yaxis.set_major_formatter(fmt_int)
plt.xlabel("Length (bytes)")
plt.tight_layout()
plt.savefig(OUTDIR / "length_hist_logy.png", dpi=150)

# 5) Per-day counts line chart

plt.figure(figsize=(14, 5))
ax = daily.plot(kind="line", title="Per-Day Message Volume")
ax.set_xlabel("Date")
ax.set_ylabel("Count")
bar_no_sci(ax)

# format ticks as YYYY-MM-DD
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
plt.gcf().autofmt_xdate(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTDIR / "per_day_counts.png", dpi=150)
