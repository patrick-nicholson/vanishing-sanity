import numpy as np


def weighted_percentile(data, weights, perc):
    """
    perc : percentile in [0-1]!
    https://stackoverflow.com/a/61343915
    """
    ix = np.argsort(data)
    data = np.asarray(data)[ix]
    weights = np.asarray(weights)[ix]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)  # 'like' a CDF function
    return np.interp(perc, cdf, data)
