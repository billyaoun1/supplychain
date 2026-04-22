import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

data = [
    ["C1", 180, 250, 5, 12, 108, 105, 40, 82, 3, 2, 10],
    ["C2", 420, 1200, 3, 18, 115, 118, 58, 74, 6, 5, 18],
    ["C3", 95, 100, 7, 8, 104, 101, 32, 88, 2, 1, 7],
    ["C4", 760, 2800, 2, 24, 121, 130, 72, 69, 8, 7, 28],
    ["C5", 510, 1800, 3, 20, 117, 123, 64, 72, 7, 6, 22],
    ["C6", 140, 400, 6, 10, 106, 104, 36, 85, 3, 2, 9],
    ["C7", 300, 900, 4, 15, 111, 112, 49, 79, 5, 4, 14],
    ["C8", 880, 3500, 2, 28, 125, 138, 80, 65, 9, 8, 32],
    ["C9", 210, 600, 5, 13, 109, 108, 42, 81, 4, 3, 11],
    ["C10", 640, 2200, 2, 22, 119, 127, 69, 70, 8, 6, 26],
    ["C11", 160, 300, 6, 9, 105, 103, 35, 86, 3, 2, 8],
    ["C12", 390, 1400, 4, 17, 114, 116, 55, 76, 6, 5, 17]
]

columns = [
    "Case", "Package Value ($k)", "Supplier Distance (mi)", "# Suppliers", 
    "Commodity Volatility", "Inflation Index", "Freight Cost Index", 
    "Port Congestion Index", "Supplier Reliability", "Fabrication Complexity", 
    "Customization", "Historical Lead Time (weeks)"
]

df = pd.DataFrame(data, columns=columns)

df.set_index('Case', inplace=True)

scaler = StandardScaler()

standardized_matrix = scaler.fit_transform(df)

df_standardized = pd.DataFrame(standardized_matrix, columns=df.columns, index=df.index)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(df_standardized)

"""
pca = PCA(n_components=2) 

# 2. Fit the model and transform your standardized data
principal_components = pca.fit_transform(df_standardized)

# 3. Put it into a nice dataframe
df_pca = pd.DataFrame(data=principal_components, 
                      columns=['Principal Component 1', 'Principal Component 2'], 
                      index=df.index)

print(df_pca)
"""

cov_matrix = np.cov(df_standardized.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
top_2_eigenvectors = sorted_eigenvectors[:, 0:2]

manual_pca_scores = np.dot(df_standardized, top_2_eigenvectors)
print(manual_pca_scores)

pc1 = manual_pca_scores[:, 0]
pc2 = manual_pca_scores[:, 1]
explained_var = eigenvalues[sorted_indices][:2] / eigenvalues.sum()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pc1, pc2, s=80, c="steelblue", edgecolors="black", zorder=3)

for case, x, y in zip(df.index, pc1, pc2):
    ax.annotate(case, (x, y), textcoords="offset points", xytext=(7, 5), fontsize=9)

ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
ax.set_title("PCA: Cases projected onto first two principal components")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("pca_plot.png", dpi=150)
plt.show()