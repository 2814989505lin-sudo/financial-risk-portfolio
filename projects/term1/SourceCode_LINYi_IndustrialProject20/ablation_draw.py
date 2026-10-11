import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = "ablation_rgcn_depth_mlp_fast.csv"
df = pd.read_csv(csv_path)

print(df)

df["mlp_hidden_str"] = df["mlp_hidden"].astype(str)

mlp_order = ["[]", "[64]", "[64, 64]", "[64, 64, 64]"]
mlp_labels = {
    "[]": "No MLP",
    "[64]": "MLP [64]",
    "[64, 64]": "MLP [64, 64]",
    "[64, 64, 64]": "MLP [64, 64, 64]",
}

plt.figure(figsize=(7, 5))

for key in mlp_order:
    sub = df[df["mlp_hidden_str"] == key].sort_values("gnn_layers")
    if sub.empty:
        continue
    plt.plot(
        sub["gnn_layers"],
        sub["best_val_macro_f1"],
        marker="o",
        label=mlp_labels.get(key, key),
    )

best_idx = df["best_val_macro_f1"].idxmax()
best_row = df.loc[best_idx]

plt.scatter(
    best_row["gnn_layers"],
    best_row["best_val_macro_f1"],
    s=80,
)
plt.text(
    best_row["gnn_layers"] + 0.05,
    best_row["best_val_macro_f1"],
    f"Best: {int(best_row['gnn_layers'])} layers + {best_row['mlp_hidden']}\nF1={best_row['best_val_macro_f1']:.3f}",
    fontsize=9,
    va="bottom",
)

plt.xlabel("Number of R-GCN layers")
plt.ylabel("Best validation macro F1")
plt.title("Ablation on R-GCN depth and MLP depth")
plt.xticks([1, 2, 3])
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

output_path = "ablation_rgcn_depth_mlp_fast.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Ablation figure saved to: {output_path}")
