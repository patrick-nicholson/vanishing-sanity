from typing import Optional, Union, Callable, List

from . import sampling


class SqlSampling:
    """Expression generation for uniform-hash-based deterministic sampling
    """

    def __init__(self, hash_function: str, hash_bits: int, signed: bool, *,
                 hash_min: Optional[int] = None, hash_max: Optional[int] = None):
        """
        Parameters
        ----------
        hash_function : str
            Hash function in the dialect. Must support multiple columns for self.rnorm to work.
        hash_bits : int
            Number of bits in the hash function. Can be None if hash_min and hash_max are provided.
        signed : bool
            Does the hash return a signed integer centered at 0
        hash_min, hash_max : int, optional
            The minimum and maximum values of the hash function, respectively
        """
        if (hash_bits is None) and ((hash_min is None) or (hash_max is None)):
            raise ValueError()
        self.hash_function = hash_function
        self.hash_bits = hash_bits
        self.signed = signed
        if self.hash_bits is None:
            self.hash_min = hash_min
            self.hash_max = hash_max
        else:
            self.hash_min = -2 ** (self.hash_bits - 1) if self.signed else 0
            self.hash_max = 2 ** (self.hash_bits - 1) - 1 if self.signed else 2 ** self.hash_bits - 1
        self.hash_range = self.hash_max - self.hash_min

    def _hash(self, column: str, *columns, double: bool = False):
        """Call the hashing function"""
        cols = ", ".join([column, *columns])
        hashed = f"{self.hash_function}({cols})"
        if double:
            return f"CAST({hashed} AS DOUBLE)"
        return hashed

    def runif_column(self, column: str, *columns):
        """Deterministic uniform number in (0, 1)"""
        hashed = self._hash(column, *columns, double=True)
        if self.signed:
            return f"(({hashed} + 1) / ({self.hash_max + 2}.0 / 2) + .5)"
        return f"({hashed} / {self.hash_max + 1}.0)"

    def _rpois_sequence_otherwise(self, lam: float, tolerance: Optional[float] = None,
                                  max_length: Optional[int] = None):
        return sampling.RandomPoissonIntegerRanges \
            .poisson_ranges(self.hash_min, self.hash_max, lam=lam, tolerance=tolerance,
                            max_length=max_length)

    def rpois_column(self, column: str, *columns: str, lam: float,
                     tolerance: Optional[float] = None, max_length: Optional[int] = None):
        """Deterministic random Poisson by sequential search"""
        sequence, otherwise = self._rpois_sequence_otherwise(lam, tolerance, max_length)
        hashed = self._hash(column, columns)
        whens = [f"WHEN {hashed} < {right_int} THEN {deviate}" for right_int, deviate in sequence]
        return f"(CASE {' '.join(whens)} ELSE {otherwise} END)"

    def bootstrap_weight_column(self, column: str, *columns: str):
        """Deterministic bootstrap weight by Poisson(1) sequential search"""
        return self.rpois_column(column, *columns, 1.)

    def rnorm_column(self, column: str, *columns):
        """Deterministic random normal by Irwin-Hall approximation

        Reference
        ---------
        https://en.wikipedia.org/wiki/Irwin%E2%80%93Hall_distribution#Approximating_a_Normal_distribution

        Notes
        -----
        This differs from the linked derivation in its use of random uniform integers.
        If unsigned, the mean of each integer is zero so centering the sum is not required.
        The sum simply needs to be scaled into the normal range.
        """
        if not self.signed:
            raise NotImplementedError()
        summation = " + ".join(
            f"{self._hash({i}, column, *columns, double=True)} / {self.hash_range}.0"
            for i in range(12)
        )
        return f"({summation})"

    def sample(self, table_name: str, frac: float, replace: bool, column: str, *columns: str,
               array_range: Optional[Union[Callable[[int], str], str]] = None,
               array_flatten: Optional[str] = None, weight_alias: str = "__sample_weight",
               index_alias: str = "__sample_index"):
        """
        Deterministic sampling with or without replacement

        Sampling with replacement requires the vendor to support:
        - subquery
        - arrays
        - either array literals or creating an array representing a range
        - array flattening (creating multiple rows by placing each value in an array on a new row)

        Parameters
        ----------
        array_range : callable or str, required if replace==True
            Function taking an integer that returns an array of that size containing positional
            index values, or the name of a vendor-specific function that does the same.
            - If callable, a large CASE statement will be generated to map the resampling weight to
            an array literal.
            - If string, the database function must support dynamic creation
        array_flatten : str, required if replace==True
            Name of a vendor-specific function that flattens/explodes an array so that a row is
            repeated allow the length of the array.

        Notes
        -----
        For Postgres, array_range=postgres_array_constructor (see below) and array_flatten='unnest'.
        For Spark/Hive, array_range='array_range' and array_flatten='explode'.
        """
        sampling.validate_fraction(frac=frac, replace=replace)
        if not replace:
            return self._sample_without_replacement(table_name, frac, column, *columns)
        if (array_range is None) or (array_flatten is None):
            raise ValueError(
                "array_range and array_flatten must be provided for sampling with replacement")
        return self._sample_with_replacement(
            table_name, frac, [column, *columns], array_range, array_flatten, weight_alias,
            index_alias)

    def _sample_without_replacement(self, table_name: str, frac: float, column: str, *columns: str):
        threshold = int(self.hash_min + frac * self.hash_range)
        hashed = self._hash(column, *columns)
        return f"SELECT * FROM {table_name} WHERE {hashed} < {threshold}"

    def _sample_with_replacement(self, table_name: str, frac: float, columns: list[str],
                                 array_range: Union[Callable[[int], str], str], array_flatten: str,
                                 weight_alias: str, index_alias: str):
        weight = self.rpois_column(*columns, lam=frac)
        subquery = f"SELECT *, {weight} AS {weight_alias} FROM {table_name}"
        if callable(array_range):
            _, otherwise = self._rpois_sequence_otherwise(frac, None, None)
            whens = " ".join(f"WHEN {weight_alias} = i THEN {array_range(i)}" for i in range(1, otherwise + 1))
            array = f"CASE {whens} END"
        else:
            array = f"{array_range}({weight_alias})"
        return f"""
            SELECT *, {array_flatten}({array}) AS {index_alias}
            FROM ({subquery}) sampling_weights
            WHERE 0 < {weight_alias}
        """


def postgres_array_range(size: int) -> str:
    """Array range function for Postgres"""
    return f"ARRAY[{','.join(str(i) for i in range(size))}]"
