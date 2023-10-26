import numpy as np
from typing import Union
from scipy.stats import normaltest, levene, ttest_ind, f_oneway


def check_normality(sample: np.array) -> bool:
    p_value = normaltest(sample)[1]
    is_normal = p_value > 0.05
    return is_normal


def check_homoscedasticity(*samples: np.array) -> bool:
    p_value = levene(*samples)[1]
    is_homoscedastic = p_value > 0.05
    return is_homoscedastic


def check_group_difference(*samples: np.array) -> Union[bool, None]:
    is_significant = None
    if len(samples) == 2:
        p_value = ttest_ind(*samples)[1]
    elif len(samples) > 2:
        p_value = f_oneway(*samples)[1]
    is_significant = p_value < 0.05
    return is_significant
