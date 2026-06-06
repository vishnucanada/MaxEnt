"""Small compatibility shims across dependency versions."""

import numpy as np

# NumPy 2.0 renamed `np.trapz` -> `np.trapezoid` and removed the old name.
# Use whichever exists so the code runs on NumPy 1.x and 2.x.
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
