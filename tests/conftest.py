"""Shared pytest fixtures. Force a non-interactive Matplotlib backend."""

import matplotlib

matplotlib.use("Agg")
