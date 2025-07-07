"""
Mass–Radius Theoretical Models Loader
-------------------------------------
Provides a unified way to load and retrieve theoretical mass–radius relationship curves.

Author: S. Wittmann
Repository: https://github.com/SimonWtmn/Exoplot
"""

import pandas as pd
from pathlib import Path

# ===========================================================
# Configuration
# ===========================================================

MODELS_DIR = Path(__file__).parent / "theoretical_models"

# Key => (filename, human-readable label)
MODEL_CATALOG = {
    "zeng_rocky": ("zeng_2019_pure_rock", "Zeng+2019: Pure Rock"),
    "zeng_iron": ("zeng_2019_pure_iron", "Zeng+2019: Pure Iron"),
    "zeng_earth": ("zeng_2019_earth_like", "Zeng+2019: Earth-like"),
    "zeng_2016_20fe": ("zeng_2016_20_Fe", "Zeng+2016: 20% Iron"),
    "Water World": ("MR-Water20_650K_DORN.txt", "Water World: 650K"),
    "marcus_collision": ("marcus_2010_maximum_collision_stripping", "Marcus+2010: Collision"),
    # ... [les autres modèles inchangés pour la lisibilité]
    # Voir ton code original pour le reste des entrées (zeng_50h2o_*, zeng_100h2o_*, etc.)
}
MODEL_CATALOG.update({
    f"zeng_{pct}h2_{temp}K": (
        f"zeng_2019_{pct}_H2_onto_earth_like_{temp}K",
        f"Zeng+2019: {pct}% H₂ @ {temp}K"
    )
    for pct in [0.1, 0.3, 1, 2, 5]
    for temp in [300, 500, 700, 1000, 2000]
})
MODEL_CATALOG.update({
    f"zeng_{pct}h2o_{temp}K": (
        f"zeng_2019_{pct}_H2O_{temp}K",
        f"Zeng+2019: {pct}% H₂O @ {temp}K"
    )
    for pct in [50, 100]
    for temp in [300, 500, 700, 1000]
})


# ===========================================================
# Functions
# ===========================================================

def get_model_curve(key: str) -> pd.DataFrame:
    """
    Load the mass–radius curve for a given model key.
    Returns a DataFrame with columns ['mass', 'radius'].
    """
    if key not in MODEL_CATALOG:
        raise KeyError(f"Invalid model key '{key}'. Use list_models() to view available options.")

    filename = MODEL_CATALOG[key][0]
    filepath = MODELS_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    df = pd.read_csv(filepath, sep=r'\s+|\t+', header=None, engine='python')
    if df.shape[1] != 2:
        raise ValueError(f"Model file '{filename}' should have exactly 2 columns.")
    df.columns = ['mass', 'radius']
    return df.dropna()


def get_model_label(key: str) -> str:
    """Return the human-readable label for a model key."""
    return MODEL_CATALOG.get(key, (None, key))[1]


def list_models() -> dict:
    """
    Return the full model catalog: key => (filename, label).
    """
    return MODEL_CATALOG.copy()
