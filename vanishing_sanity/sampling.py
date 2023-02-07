from functools import partial
from math import exp
from typing import Optional

from scipy.stats import poisson

INT32_MIN = -2 ** 31
INT32_MAX = 2 ** 31 - 1
INT64_MIN = -2 ** 63
INT64_MAX = 2 ** 63 - 1

DEFAULT_POIS_TOL = 1e-6
DEFAULT_POIS_MAX_LEN = 100


def rpois_check(*, lam, tolerance, max_length):
    """Check sequential search parameters are valid"""
    if (lam is None) or (lam <= 0):
        raise ValueError("Must provide a positive value lam (Poisson lambda)")
    max_value = int(poisson.isf(tolerance, lam)) + 1
    if max_length < max_value:
        raise ValueError(
            f"Too many values {max_value} for lambda={lam} and tolerance={tolerance}. "
            "Decrease `tolerance`, increase `max_length`, or use a more efficient method.")
    return max_value


class RandomPoissonIntegerRanges:

    @staticmethod
    def poisson_ranges(integer_min: int, integer_max: int, lam: float,
                       tolerance: Optional[float] = None, max_length: Optional[int] = None):
        """Determines the Poisson uniform integer range

        The return value is an array of the upper values for a Poisson draw for the 0-based index
        of that value. i.e., if uniform int < ranges[0] then 0, ...

        Reference
        ---------
        https://en.wikipedia.org/wiki/Poisson_distribution#Random_drawing_from_the_Poisson_distribution

        Notes
        -----
        This differs from the linked derivation in its use of random uniform integers. The only
        required change is switching from CDF ranges in [0, 1] to integer ranges in
        [integer_min, integer_max], the range of a uniform hashing function (typically the 32-bit or
        64-bit min and max values, respectively).
        """
        if tolerance is None:
            tolerance = DEFAULT_POIS_TOL
        if max_length is None:
            max_length = DEFAULT_POIS_MAX_LEN

        max_value = rpois_check(lam=lam, tolerance=tolerance, max_length=max_length)
        integer_range = integer_max - integer_min

        prob = exp(-lam)
        left_int = integer_min
        right_int = left_int + integer_range * prob
        tol_int = max(integer_range * tolerance, 1)

        sequence = list()
        deviate = 0
        while (deviate < max_value) and (right_int < integer_max):
            if tol_int < right_int - left_int:
                sequence.append((int(right_int), deviate))
                left_int = right_int
            elif sequence and (right_int < 0):
                sequence[-1] = (int(right_int), deviate)
            deviate += 1
            prob *= lam / deviate
            right_int += integer_range * prob

        return sequence, sequence[-1][1] + 1

    def poisson_ranges_32bit(self, lam, tolerance=None, max_length=None):
        return self.poisson_ranges(INT32_MIN, INT32_MAX, lam, tolerance, max_length)

    def poisson_ranges_64bit(self, lam, tolerance=None, max_length=None):
        return self.poisson_ranges(INT64_MIN, INT64_MAX, lam, tolerance, max_length)


def validate_fraction(*, frac, replace):
    """Check sampling fraction"""
    if (frac is None) or (frac <= 0):
        raise ValueError("Must provide a frac (fraction) argument > 0")
    if (1 <= frac) and not replace:
        raise ValueError("Cannot use frac >= 1 without replacement")
