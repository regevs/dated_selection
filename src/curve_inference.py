import scipy.special
import scipy.optimize
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tqdm

#
# Inverse coupon collector math
#
def binomln(n, k):
    # Assumes binom(n, k) >= 0
    return -scipy.special.betaln(1 + n - k, 1 + k) - np.log(n + 1)

def F(N, m, i): 
    return binomln(N, i) - m * np.log(N)

def inverse_coupon_collector(n_balls, n_unique, max_allowed):
    res = scipy.optimize.minimize_scalar(
        fun = lambda x: -F(x, n_balls, n_unique), 
        method = "bounded",
        bounds = (n_unique, max_allowed),
        options = {"xatol": 1e-3},
    )
    return res

def inverse_coupon_collector_array(n_balls_array, n_unique_array, max_allowed):
    if len(n_unique_array) == 0:
        return np.nan
    
    res = scipy.optimize.minimize_scalar(
        fun = lambda x: -F(x, n_balls_array, n_unique_array).sum(), 
        method = "bounded",
        bounds = (n_unique_array.max(), max_allowed),
        options = {"xatol": 1e-3},
    )
    return res.x

def infer_Ne(
    all_n_lineages, 
    grid_points,
    n_jobs=-1,
    max_allowed = 1e6,
):
    if not np.all(grid_points[:-1] < grid_points[1:]):
        print("grid should be sorted")
        return

    # Make sure 0 is in the grid
    if grid_points[0] != 0:
        grid_points = np.concatenate([[0], grid_points])

    # Very last grid point cannot end since we need one extra
    if grid_points[-1] == all_n_lineages.shape[1]:
        grid_points[-1] -= 1

    estimates = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
        joblib.delayed(inverse_coupon_collector_array)(
            all_n_lineages[:, grid_points[i]:grid_points[i+1]], 
            all_n_lineages[:, grid_points[i]+1:grid_points[i+1]+1], 
            max_allowed
            ) \
                for i in range(len(grid_points) - 1)
    ) 
    # estimates = [inverse_coupon_collector_array(
    #         all_n_lineages[:, grid_points[i]:grid_points[i+1]], 
    #         all_n_lineages[:, grid_points[i]+1:grid_points[i+1]+1], 
    #         max_allowed
    #         ) \
    #             for i in tqdm.trange(len(grid_points) - 1)
    # ]
        
    estimates = np.array(estimates)
    estimates[estimates > max_allowed*0.9] = np.nan

    return estimates

    
# Read all n_lineage files from a directory into a single array
def read_all_n_lineages(path):    
    all_filenames = list(Path(path).glob("*/n_lineages.csv.gz"))

    readme = lambda filename: pd.read_csv(filename, header=None).values
    all_n_lineages = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(readme)(filename) for filename in all_filenames
    )
                    
    all_n_lineages = np.concatenate(all_n_lineages, axis=0)
    return all_n_lineages
