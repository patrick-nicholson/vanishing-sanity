"""Spark helpers
"""
import pyspark.sql.functions as F
import pyspark.sql.types as T

from functools import lru_cache, wraps
from pyspark.sql import Column, DataFrame
from types import SimpleNamespace

from . import sampling


def add_tab_completion(property_name: str = "k"):
    """Add tab completion for columns to a Spark DataFrame as a new property"""
    @lru_cache
    def namespace(self):
        """Wraps columns in a SimpleNamespace for tab completion"""
        return SimpleNamespace(**{k: self[k] for k in self.columns})
    from pyspark.sql import DataFrame
    setattr(DataFrame, property_name, property(namespace))


def add_alias(func):
    """Alias the result of a function that returns a Spark column with the function signature"""
    @wraps(func)
    def inner(*args, **kwargs):
        params = [str(x) for x in args]
        params.extend(f"{k}={v}" for k, v in kwargs.items())
        alias = f"""{func.__name__}({", ".join(params)})"""
        return func(*args, **kwargs).alias(alias)
    return inner


def as_column(x):
    """Convert to column"""
    if isinstance(x, Column):
        return x
    if isinstance(x, str):
        return F.col(x)
    raise ValueError(f"Unsupported type {type(x)}")


def recode_column(column, mapping, otherwise=None):
    """Recode the values of a column"""
    column = as_column(column)
    if hasattr(mapping, "items"):
        mapping = mapping.items()
    result = None
    for old_value, new_value in mapping:
        result = getattr(result, "when", F.when)(column == old_value, new_value)
    if otherwise is not None:
        result = result.otherwise(otherwise)
    return result


def decimal_conversion(df, *, subset=None):
    """Convert DecimalType to a primitive type inferred from decimal scale."""
    decimals = {
        field.name: field.dataType
        for field in df.schema.fields
        if isinstance(field.dataType, T.DecimalType)
        if (subset is None) or (field.name in subset)
    }
    new_type = {
        name: T.DoubleType() if type_.scale else T.LongType()
        for name, type_ in decimals.items()
    }
    return df.select(*[
        column if column not in new_type
        else F.col(column).cast(new_type[column]).alias(column)
        for column in df.columns
    ])


@add_alias
def runif(column, *columns):
    """Deterministic random uniform value"""
    return (F.hash(column, *columns).cast("double") + 1) / ((2**31 + 1) / 2) + .5


@add_alias
def rpois(column, *columns, lam=None, tolerance=None, max_length=None):
    """Deterministic random Poisson by sequential search"""
    sequence, otherwise = sampling.RandomPoissonIntegerRanges\
        .poisson_ranges_32bit(lam, tolerance, max_length)
    uniform_int = F.hash(column, *columns)
    result = None
    for right_int, deviate in sequence:
        result = getattr(result, "when", F.when)(uniform_int < right_int, deviate)
    result = result.otherwise(otherwise)
    return result


@add_alias
def poisson_bootstrap(column, *columns):
    """Deterministic Poisson bootstrap by lambda=1 sequential search"""
    return rpois(column, *columns, lam=1, tolerance=1e-12)


@add_alias
def rnorm(column, *columns):
    """Deterministic random normal by Irwin-Hall approximation

    Reference
    ---------
    https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution#Approximating_a_Normal_distribution

    Notes
    -----
    This differs from the linked derivation in its use of random uniform integers. The mean of each
    integer is zero so centering the sum is not required. The sum simply needs to be scaled into the
    normal range.
    """
    return sum(F.hash(F.lit(i), column, *columns) / 2**32 for i in range(12))


@add_alias
def sample(df, column, *columns, frac=None, replace=False, tolerance=None, max_length=None):
    """Deterministically sample from a DataFrame with or without replacement

    Note
    ----
    Data are not shuffled
    """
    sampling.validate_fraction(frac=frac, replace=replace)
    if not replace:
        return df.filter(runif(column, *columns) < frac)
    k = "__poisson_boostrap__"
    rp = rpois(column, *columns, lam=frac, tolerance=tolerance, max_length=max_length)
    return (
        df.withColumn(k, rp)
        .withColumn(k, F.sequence(F.lit(0), F.col(k)))
        .withColumn(k, F.explode(k))
        .filter(F.col(k) > 0)
        .drop(k)
    )
