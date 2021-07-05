"""
Some metrics for beta distribution. WIP, because it only supports pd.Series now.
"""

from scipy import stats as ss
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Union, List


BETA_PRIOR_TYPE = Optional[
    Tuple[Union[float, pd.Series], Union[float, pd.Series]]
] # (prior convs, prior non_convs)


def get_alpha_and_beta(convs: pd.Series, clicks: pd.Series, 
                        prior: BETA_PRIOR_TYPE = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given convs and clicks get the a posteriori parameters (alpha and beta) for
    Beta distribution of possible values of CR.

    Args:
        prior: prior pseudo-observations for Beta distribution. If None, then we assume
            Beta(0, 0) prior (we still need to deal with zeros as parameters)
    """
    
    non_convs = clicks - convs
    
    if prior is None: # corresponds to Beta(0, 0) prior
        # beta is ill-defined for 0
        # I checked empirically that the right end of the interval is ok if we replace 0 with 1
        # however we need to replace the left end of the interval with zero for convs == 0
        alpha = convs.mask(convs <= 0, 1)
        beta = non_convs.mask(non_convs <= 0, 1)
    else:
        alpha = convs.clip(lower=0) + prior[0]
        beta = non_convs.clip(lower=0) + prior[1]

    return alpha, beta


def get_beta_intervals(convs: pd.Series, clicks: pd.Series,
                        alpha: float = 0.95, 
                        prior: BETA_PRIOR_TYPE = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    For given convs and clicks return left and right ends of confidence intervals containg
    alpha percent of Beta distribution Beta(alpha, beta), where
        alpha, beta = get_alpha_and_beta(convs, clicks, prior=prior)

    For the given observations we'd like to get bounds for possible values of CR that
    generated the data.

    Args:
        alpha: float, 0 < alpha < 1
        prior: same as in get_alpha_and_beta
    """
    modified_convs, non_convs = get_alpha_and_beta(convs, clicks, prior=prior)
    
    left, right = ss.beta.interval(alpha, modified_convs, non_convs)
    # special case of zero convs - set left as zero - regardless of the prior
    left = left * (convs > 0)
    return left, right


def resample_CR(convs: pd.Series, clicks: pd.Series, 
                prior: BETA_PRIOR_TYPE = None) -> np.ndarray:
    """
    Resample CR from the a posteriori Beta distribution.
    For convs == 0, in case no prior is given, we set 
    resampled_cr = get_minimal_viable_CR(modified_convs, non_convs).
    """
    modified_convs, non_convs = get_alpha_and_beta(convs, clicks, prior=prior)
    
    resampled_cr = np.random.beta(modified_convs, non_convs)
    
    if prior is not None:
        return resampled_cr

    # special case of zero convs in case of no prior
    min_cr = get_minimal_viable_CR(modified_convs, non_convs)
    return np.where(convs > 0, resampled_cr, min_cr)


def get_better_CR(convs: pd.Series, clicks: pd.Series,
                 prior: BETA_PRIOR_TYPE = None) -> np.ndarray:
    """
    CR calculated with consideration of a posteriori Beta distribution. 
    For convs == 0, in case no prior is given, we set 
    better_CR = get_minimal_viable_CR(modified_convs, non_convs).
    """
    
    modified_convs, non_convs = get_alpha_and_beta(convs, clicks, prior=prior)
    
    mean_cr = ss.beta.mean(modified_convs, non_convs)
    
    if prior is not None:
        return mean_cr
    
    # special case of zero convs in case of no prior
    min_cr = get_minimal_viable_CR(modified_convs, non_convs)
    return np.where(convs > 0, mean_cr, min_cr)


def get_minimal_viable_CR(alpha: pd.Series, beta: pd.Series) -> np.array:
    """
    For given Beta-distribution return left end of 0.95 confidence interval
    - interpreted as minimum practically possible value of CR for the distribution
    """
    return ss.beta.interval(0.95, alpha, beta)[0]


def get_interval_metrics(predictions: pd.Series, targets: pd.Series,
                            left: pd.Series,
                            right: pd.Series) -> pd.DataFrame:
    
    df = pd.DataFrame()

    df["below"] = predictions < left
    df["above"] = predictions > right
    df["inside"] = (left <= predictions) & (predictions <= right)
    
    # weights for the loss function taking into account the intervals
    # weight is equal to 1 if prediction is outside of the interval
    # and scales linearly to 0 towards the targets
    inside_and_below_target_weight = (targets - predictions) / (targets - left)
    inside_and_above_target_weight = (predictions - targets) / (right - targets)
    inside_weight = np.where(predictions < targets, inside_and_below_target_weight, inside_and_above_target_weight)
    df["weights"] = np.where(df["inside"], inside_weight, 1)

    return df
