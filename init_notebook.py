
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import pymc as pm
import xarray as xr
import arviz as az
import utils as utils

from scipy import stats as stats
from matplotlib import pyplot as plt


# ---
# Print imports / aliases
with open(__file__) as f:
    lines = f.readlines()

from colorama import Fore
def print_import(import_line):
    if len(import_line.strip()) > 1:
        parts = [p for p in import_line.split(" ") if p]
        if parts[0] == 'import':
            module = parts[1]
            alias = parts[3]
            msg = Fore.GREEN + 'import' \
                + Fore.BLUE + f" {module} " \
                + Fore.GREEN + "as" \
                + Fore.BLUE + f" {alias}"
        elif parts[0] == 'from':
            module = parts[1]
            submodule = parts[3]
            alias = parts[5]
            msg = Fore.GREEN + 'from' \
                + Fore.BLUE + f" {module} "\
                + Fore.GREEN + 'import' \
                + Fore.BLUE + f" {submodule} " \
                + Fore.GREEN + "as" \
                + Fore.BLUE + f" {alias}"
        print(msg)

print(Fore.RED + f"Module aliases imported by init_notebook.py:\n{'-'* 44}")
for l in lines:
    if "# ---" in l:
        break
    print_import(l)

from watermark import watermark
print(Fore.RED + f"Watermark:\n{'-'* 10}")
print(Fore.BLUE + watermark())
print(Fore.BLUE + watermark(iversions=True, globals_=globals()))


# Keep the lectures' 94% highest-density intervals explicit. ArviZ 1 defaults
# to equal-tail intervals and a different probability.
az.rcParams["stats.ci_kind"] = "hdi"
az.rcParams["stats.ci_prob"] = 0.94
az.rcParams["stats.round_to"] = 3

from matplotlib import style
from pathlib import Path
STYLE = Path(__file__).resolve().with_name("statistical-rethinking-2023.mplstyle")
style.use(STYLE)
