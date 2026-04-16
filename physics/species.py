import csv
from pathlib import Path

SPECIES_DB = Path(__file__).parent / "solutes.csv"

_FLOAT_FIELDS = {
    "molar_volume", "nu", "A_arrhenius", "Ea", "theta", "eta", "SA",
    "A1", "A2", "A3", "A4", "A5", "A6"
}


def load_species(name):
    """Load a single species by name from solutes.csv.

    Returns a dict with all parameters as floats.
    Raises KeyError if the species is not found.
    """
    with open(SPECIES_DB, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["name"].strip().lower() == name.strip().lower():
                return {
                    k: (float(v) if k in _FLOAT_FIELDS else v.strip())
                    for k, v in row.items()
                }
    raise KeyError(f"Species '{name}' not found in {SPECIES_DB}")