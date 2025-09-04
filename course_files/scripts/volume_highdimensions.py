import numpy as np
import matplotlib.pyplot as plt

# Fraction of volume within epsilon of the boundary
# - Unit d-cube [0,1]^d:    f_cube(d, ε)   = 1 - (1 - 2ε)^d
# - Unit d-ball (radius 1): f_ball(d, ε)   = 1 - (1 - ε)^d
#   (ε is a shell thickness; inner radius = 1 - ε)

dims = np.arange(1, 201)  # dimensions 1..200
epsilons = [0.05, 0.1]

plt.figure(figsize=(7, 5))
for eps in epsilons:
    cube_frac = 1 - (1 - 2*eps)**dims
    ball_frac = 1 - (1 - eps)**dims
    plt.plot(dims, cube_frac, lw=2, label=f"Cube, ε={eps}")
    plt.plot(dims, ball_frac, lw=2, ls="--", label=f"Ball, ε={eps}")

plt.title("Fraction of volume within ε of the boundary vs. dimension")
plt.xlabel("Dimension (d)")
plt.ylabel("Fraction near boundary")
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.show()