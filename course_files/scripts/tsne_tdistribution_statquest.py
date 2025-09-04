import numpy as np
import matplotlib.pyplot as plt
from math import gamma, sqrt, pi

def student_t_pdf(x: np.ndarray, df: float) -> np.ndarray:
    coeff = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2))
    return coeff * (1 + (x**2) / df) ** (-(df + 1) / 2)

# --- Plot 1: Student's t-distribution ---
x = np.linspace(-5, 5, 800)
df = 3
pdf = student_t_pdf(x, df)

plt.figure(figsize=(6, 4))
plt.plot(x, pdf, color="tab:blue", lw=2)
plt.title(f"Student's t-distribution (df={df})")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True, alpha=0.3)

# --- Plot 2: Four dispersed 2D clusters ---
np.random.seed(0)
points_per_cluster = 150
means = np.array([
    [ 4.0,  3.0],
    [-4.0, -3.0],
    [ 3.5, -2.5],
    [-3.5,  2.5],
])
# Slightly elongated and rotated covariances
covs = [
    np.array([[0.6, 0.25],[0.25, 0.4]]),
    np.array([[0.5, -0.2],[-0.2, 0.5]]),
    np.array([[0.4, 0.15],[0.15, 0.3]]),
    np.array([[0.5, -0.25],[-0.25, 0.35]]),
]

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
labels = [f"Cluster {i+1}" for i in range(4)]

plt.figure(figsize=(6, 6))
for mean, cov, color, label in zip(means, covs, colors, labels):
    pts = np.random.multivariate_normal(mean, cov, size=points_per_cluster)
    plt.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.75, edgecolor="k", linewidths=0.2, c=color, label=label)

plt.title("Four dispersed clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, loc="best")

plt.show()