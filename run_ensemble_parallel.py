import os
import numpy as np
import time
import pandas as pd
import datetime as dt
from concurrent.futures import ProcessPoolExecutor, as_completed

num_cores = 50


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


def process_simulation(j, df, ssps, pmat, nt, nssps):
    """
    Process a single simulation for a given ensemble member.

    Parameters:
    ----------
    j : int
        Index of the ensemble member.
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
    T2mat = np.zeros((nt, len(df), nssps))
    srmmat = np.zeros((nt, len(df), nssps))
    demat = np.zeros((nt, len(df), nssps))
    T0mat = np.zeros((nt, len(df), nssps))

    for i, n in enumerate(df.name):
        for k, ssp in enumerate(ssps[:]):
            (
                Cmat[:, i, k],
                Fmat[:, i, k],
                Tmat[:, i, k],
                T2mat[:, i, k],
                srmmat[:, i, k],
                demat[:, i, k],
                T0mat[:, i, k],
            ) = adpt_fair(ssp, 5, 1.5, df, i=i, p=pmat.iloc[[j]], iters=50)

    return j, Cmat, Fmat, Tmat, T2mat, srmmat, demat, T0mat


if __name__ == "__main__":
    pmat = pd.read_json("params.json")
    nens = len(pmat[:])
    nt = len(ssp3ext.Emissions.emissions)
    nssps = len(ssps)

    # Initialize arrays to store results
    Cmat = np.zeros((nt, len(df), nens, nssps))
    Fmat = np.zeros((nt, len(df), nens, nssps))
    Tmat = np.zeros((nt, len(df), nens, nssps))
    T2mat = np.zeros((nt, len(df), nens, nssps))
    srmmat = np.zeros((nt, len(df), nens, nssps))
    demat = np.zeros((nt, len(df), nens, nssps))
    T0mat = np.zeros((nt, len(df), nens, nssps))

    t0 = time.time()

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [
            executor.submit(process_simulation, j, df, ssps, pmat, nt, nssps)
            for j in range(nens)
        ]

        # Track progress
        for i, future in enumerate(as_completed(futures), start=1):
            j, Cmat_j, Fmat_j, Tmat_j, T2mat_j, srmmat_j, demat_j, T0mat_j = (
                future.result()
            )

            # Store results for the current ensemble member
            Cmat[:, :, j, :] = Cmat_j
            Fmat[:, :, j, :] = Fmat_j
            Tmat[:, :, j, :] = Tmat_j
            T2mat[:, :, j, :] = T2mat_j
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
        T2mat=T2mat,
        srmmat=srmmat,
        demat=demat,
        T0mat=T0mat,
    )

    print("Simulation results saved to 'output/simulation_results.npz'")
