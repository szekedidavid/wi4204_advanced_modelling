import csv
from pathlib import Path

SPECIES_DB = Path(__file__).parent / "solutes.csv"

_FLOAT_FIELDS = {"molar_volume", "A_arrhenius", "Ea", "SA"}
_NU_PREFIX    = "nu_"


def load_species(name):
    with open(SPECIES_DB, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["name"].strip().lower() == name.strip().lower():
                return {
                    k: (float(v) if k in _FLOAT_FIELDS else v.strip())
                    for k, v in row.items()
                }
    raise KeyError(f"Species '{name}' not found in {SPECIES_DB}")


def load_precipitate(name, solute_names):
    with open(SPECIES_DB, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["name"].strip().lower() != name.strip().lower():
                continue
            out = {"name": name, "nu": {}}
            for k, v in row.items():
                v = v.strip()
                if k == "name":
                    continue
                if k.startswith(_NU_PREFIX):
                    sname = k[len(_NU_PREFIX):]
                    if sname in solute_names:
                        out["nu"][sname] = int(float(v))
                elif k in _FLOAT_FIELDS:
                    out[k] = float(v)
            return out
    raise KeyError(f"Precipitate '{name}' not found in {SPECIES_DB}")


def load_precipitates(names, solute_names):
    return [load_precipitate(n, solute_names) for n in names]