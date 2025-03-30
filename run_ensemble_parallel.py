# %%
import fair

# %%
import numpy as np
from scipy import stats
from scipy import signal
import pandas as pd
from matplotlib import pyplot as plt


# %%
from scipy import stats
from fair.tools.ensemble import tcrecs_generate
from fair.forward import fair_scm

# %%
from functions import *

# %%
fair._version.get_versions()["version"]

# %%
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen

from fair.tools.constrain import hist_temp


# %%
from fair.RCPs import rcp3pd, rcp45, rcp6, rcp85
from fair.SSPs import ssp370, ssp126, ssp585, ssp119, ssp245, ssp534, ssp460

# %%
samples = 400

# %%
pmat0 = make_params(n=samples)

# %%


# %%
if 0:
    Ce, Fe, Te = run_fair(pmat0, ssp585.Emissions.emissions)

    # load up Cowtan and Way data remotely
    url = "http://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_annual_v2_0_0.txt"
    response = urlopen(url)

    CW = np.loadtxt(response)
    constrained = np.zeros(samples, dtype=bool)

    for i in range(samples):
        # we use observed trend from 1880 to 2016
        constrained[i], _, _, _, _ = hist_temp(
            CW[30:167, 1], Te[1880 - 1765 : 2017 - 1765, i], CW[30:167, 0]
        )

    # How many ensemble members passed the constraint?
    print("%d ensemble members passed historical constraint" % np.sum(constrained))
    pmat = pmat0.iloc[constrained]
    pmat.to_json("params.json")
else:
    pmat = pd.read_json("params.json")


# %%
yr = ssp119.Emissions.year

# %%
pmat

# %%
np.array(pmat["F_scale"]).shape

# %%
import copy

# %%
ssp3ext = copy.deepcopy(ssp534)
ssp3ext.Emissions.emissions[340:, 1] = ssp3ext.Emissions.emissions[340, 1]
ssp3ext.Emissions.emissions[440:480, 1] = ssp3ext.Emissions.emissions[340, 1] * (
    1 - np.arange(1, 41, 1) / 40
)
ssp3ext.Emissions.emissions[480:, 1] = 0

plt.plot(ssp3ext.Emissions.emissions[:, 0], ssp3ext.Emissions.emissions[:, 1])


d = {
    "name": ["SRMPa", "SRMPb", "SRMPc", "SRMPd", "SRMPe"],
    "Start": [2040, 2040, 2040, 2040, 2040],
    "End": [2170, 2170, 2170, 2170, 2170],
    "Effic": [0.99, 0.9, 0.5, 0.7, 0.99],
    "fade": [20, 10, 5, 10, 20],
    "mhaz": [0, 0.4, 0.8, 0.4, 0.9],
    "pfail": [0.0, 0.008, 0.03, 0.015, 0.008],
    "aol": [0, 10, 10, 10, 10],
    "maxsrm": [2.0, 4.0, 4.0, 4.0, 8.0],
    "regionality": [0.1, 0.3, 0.5, 0.8, 0.1],
}
df = pd.DataFrame(data=d)
df
# %%
sgn = np.array([-1, 1, 1, 1, 1])
(1 - sgn) / 2

# %%
import numpy as np

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))

# Define the dimensions to plot
dimensions = ["Effic", "mhaz", "pfail", "maxsrm", "regionality"]
dimlong = [
    "Inability to \n meet target",
    "Mitigation\n displacement",
    "Frequency of\n interruption",
    "Max SRM \n tolerance",
    "Regionality",
]

# Number of variables
num_vars = len(dimensions)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# %%
nt = len(ssp3ext.Emissions.emissions)

# %%
ems_bs = np.sum(ssp3ext.Emissions.emissions[:, 1:3], axis=1)
Ce, Fe, T34 = run_fair(ssp3ext.Emissions.emissions, pmat0=None)

# %%
sspname = ["ssp119", "ssp245", "ssp370", "ssp460", "ssp534", "ssp585"]
ssps = [ssp119, ssp245, ssp370, ssp460, ssp534, ssp585]
nssps = len(ssps)

# %%
len(pmat)

# %%
import time
import datetime as dt

import os
import numpy as np
import time
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_simulation(j, nens, df, ssps, pmat, nt, nssps):
    """
    Process a single simulation for a given ensemble member.

    Parameters:
    ----------
    j : int
        Index of the ensemble member.
    nens : int
        Total number of ensemble members.
    df : pd.DataFrame
        DataFrame containing SRM configuration parameters.
    ssps : list
        List of SSP scenarios.
    pmat : pd.DataFrame
        Parameter matrix for the ensemble.
    nt : int
        Number of time steps.
    nssps : int
        Number of SSP scenarios.

    Returns:
    -------
    tuple
        Simulation results for the given ensemble member.
    """
    Cmat = np.zeros((nt, len(df), nssps))
    Fmat = np.zeros((nt, len(df), nssps))
    Tmat = np.zeros((nt, len(df), nssps))
    srmmat = np.zeros((nt, len(df), nssps))
    demat = np.zeros((nt, len(df), nssps))
    T0mat = np.zeros((nt, len(df), nssps))

    for i, n in enumerate(df.name):
        for k, ssp in enumerate(ssps[:]):
            (
                Cmat[:, i, k],
                Fmat[:, i, k],
                Tmat[:, i, k],
                srmmat[:, i, k],
                demat[:, i, k],
                T0mat[:, i, k],
            ) = adpt_fair(ssp, 5, 1.5, df, i=i, p=pmat.iloc[[j]], iters=50)

    return j, Cmat, Fmat, Tmat, srmmat, demat, T0mat


def calcProcessTime(starttime, cur_iter, max_iter):
    """
    Calculate elapsed and remaining time for the process.

    Parameters:
    ----------
    starttime : float
        Start time of the process.
    cur_iter : int
        Current iteration number.
    max_iter : int
        Total number of iterations.

    Returns:
    -------
    tuple
        Elapsed time, remaining time, and estimated finish time.
    """
    telapsed = time.time() - starttime
    testimated = (telapsed / cur_iter) * max_iter

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

    lefttime = testimated - telapsed  # in seconds

    return int(telapsed), int(lefttime), finishtime


def print_progress_bar(index, total, elapsed, remaining, finish_time):
    """
    Print a progress bar to the console.

    Parameters:
    ----------
    index : int
        Current progress index.
    total : int
        Total number of steps.
    elapsed : int
        Elapsed time in seconds.
    remaining : int
        Remaining time in seconds.
    finish_time : str
        Estimated finish time.
    """
    n_bar = 50  # Progress bar width
    progress = index / total
    bar = f"[{'=' * int(n_bar * progress):{n_bar}s}]"
    percentage = f"{int(100 * progress)}%"
    time_info = f"Elapsed: {elapsed}s | Remaining: {remaining}s | Finish: {finish_time}"
    print(f"\r{bar} {percentage} {time_info}", end="", flush=True)


if __name__ == "__main__":
    nens = len(pmat[:])
    nt = len(ssp3ext.Emissions.emissions)
    nssps = len(ssps)

    # Initialize arrays to store results
    Cmat = np.zeros((nt, len(df), nens, nssps))
    Fmat = np.zeros((nt, len(df), nens, nssps))
    Tmat = np.zeros((nt, len(df), nens, nssps))
    srmmat = np.zeros((nt, len(df), nens, nssps))
    demat = np.zeros((nt, len(df), nens, nssps))
    T0mat = np.zeros((nt, len(df), nens, nssps))

    t0 = time.time()

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_simulation, j, nens, df, ssps, pmat, nt, nssps)
            for j in range(nens)
        ]

        # Track progress
        for i, future in enumerate(as_completed(futures), start=1):
            j, Cmat_j, Fmat_j, Tmat_j, srmmat_j, demat_j, T0mat_j = future.result()

            # Store results for the current ensemble member
            Cmat[:, :, j, :] = Cmat_j
            Fmat[:, :, j, :] = Fmat_j
            Tmat[:, :, j, :] = Tmat_j
            srmmat[:, :, j, :] = srmmat_j
            demat[:, :, j, :] = demat_j
            T0mat[:, :, j, :] = T0mat_j

            # Update progress bar
            elapsed, remaining, finish_time = calcProcessTime(t0, i, nens)
            print_progress_bar(i, nens, elapsed, remaining, finish_time)

    print("\nAll simulations completed.")

    # Create the output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Save the output arrays
    np.savez(
        "output/simulation_results.npz",
        Cmat=Cmat,
        Fmat=Fmat,
        Tmat=Tmat,
        srmmat=srmmat,
        demat=demat,
        T0mat=T0mat,
    )

    print("Simulation results saved to 'output/simulation_results.npz'")