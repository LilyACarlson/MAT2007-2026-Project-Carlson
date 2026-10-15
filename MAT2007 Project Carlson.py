import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Read dataset
# Use your own file name and location
df = pd.read_csv("~/Desktop/MAT2007/beatles_dataset.csv")

# Choose columns to keep
df = df[["year", "album", "word_count", "lyrics"]]

# Remove unnecessary albums
df = df[((df.album != "Anthology 1") &
    (df.album != "Anthology 2") &
    (df.album != "Past Masters (Vols. 1 & 2 / Remastered)") &
    (df.album != "Rock 'n' Roll Music"))]

# Hard code years for each album
mappings = {
    "Please Please Me": 1963,
    "With the Beatles": 1963.5,
    "A Hard Day's Night": 1964,
    "Beatles for Sale": 1964.5,
    "Help!": 1965,
    "Rubber Soul": 1965.5,
    "Revolver": 1966,
    "Sgt. Pepper's Lonely Hearts Club Band": 1967,
    "Magical Mystery Tour": 1967.5,
    "The Beatles": 1968,
    "Yellow Submarine": 1969,
    "Abbey Road": 1969.5,
    "Let It Be": 1970,
}

df['year'] = df['album'].map(mappings)

# Set target words
TARGETS = {"love", "loves", "loving", "loved", "lover"}

# Count occurrences of target words
def count_targets(text):
    text = str(text).lower().replace("’", "'")
    return sum(w in TARGETS for w in re.findall(r"[a-z']+", text))


df["target_count"] = df["lyrics"].apply(count_targets)

# Calculate total count per album, in year order
per_album = (
    df.groupby(["year", "album"])["target_count"]
    .sum()
    .reset_index()
    .sort_values("year", kind="stable")
)

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(per_album["album"], per_album["target_count"], color = "mediumpurple")
ax.bar_label(bars)
ax.set_ylabel('Number of Target Words')
ax.set_xlabel("Beatles Albums in Chronological Order")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title("Total uses of words relating to love per album")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig('Beatles bar chart.png')
plt.show()

# Per-song percentage
df["song_pct"] = df["target_count"] / df["word_count"] * 100

# Per album: pooled percentage and standard deviation

per_album = (
    df.groupby(["year", "album"])
    .agg(
        target_count=("target_count", "sum"),
        word_count=("word_count", "sum"),
        sd=("song_pct", "std"),
    )
    .reset_index()
    .sort_values("year", kind="stable")
)

# Per-album percentage
per_album["pct"] = per_album["target_count"] / per_album["word_count"] * 100


# Create scatter plot with trendline, standard deviation error bars, r and p-values
x = np.arange(len(per_album))
y = per_album["pct"].to_numpy()
fit = linregress(x, y)
slope, intercept, r, p = fit.slope, fit.intercept, fit.rvalue, fit.pvalue

fig, ax = plt.subplots(figsize=(12, 6))
# ax.scatter(x, y, s=60, color = "mediumpurple")
ax.errorbar(x, y, yerr=per_album["sd"], fmt="o", markersize=7, capsize=4, label="±1 Standard Deviation", color = "mediumpurple")
ax.plot(x, slope * x + intercept, linestyle="--", color="darkturquoise")

ax.text(
    0.90, 0.75, f"r = {r:.3f}\np = {p:.3f}",
    transform=ax.transAxes,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
)

ax.set_xticks(x)
ax.set_xticklabels(per_album["album"], rotation=45, ha="right")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylabel('% of Target Words per Album')
ax.set_xlabel("Beatles Albums in Chronological Order")
ax.set_title("Use of Target Words per Beatles album")
ax.legend()
plt.tight_layout()
plt.savefig('Beatles Graph.png')
plt.show()
