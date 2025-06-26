from collections import namedtuple
from typing import Any, Union

from fuzzywuzzy import process
import numpy as np
import pandas as pd


Region = namedtuple("Region", ["code", "name", "short_name", "iso_code"])


class Regions:
    THRES = 80
    TYPES = (
        "область", "республика", "город", "край", "автономный округ", "автономная область",
        " обл", "г ", "респ ", " АО"
    )

    def __init__(self, lookup_path: str = "assets/regions.csv"):
        self._lookup_table = pd.read_csv(lookup_path, dtype=str)
        self._cache = {}
        self._index_by_code = {}
        self._index_by_name = {}
        self._index_by_short_name = {}
        self._regions = {}

        for row in self._lookup_table.itertuples():
            code = row.code
            index = row.Index
            name = row.name
            short_name = row.short_name
            iso_code = row.iso_code

            region = Region(
                code=code, name=name, short_name=short_name, iso_code=iso_code
            )
            self._regions[index] = region

            self._index_by_code[int(code)] = index
            self._index_by_short_name[short_name.lower()] = index
            self._index_by_name[name.lower()] = index

    def __len__(self) -> int:
        return len(self._regions)

    def __getitem__(self, key: Union[int, str]) -> Region:
        if isinstance(key, int) or key.isdigit():
            return self._get_by_code(key)
        elif isinstance(key, str):
            return self._get_by_name(key)
        else:
            raise TypeError("Key must me either integer/string of digits"
                            " (for getting region by code)"
                            " or string (for getting region by name)")

    def __str__(self) -> str:
        return f"{len(self)} regions of Russia"

    def _get_by_code(self, key: str) -> Region:
        index = self._index_by_code.get(int(key))

        if index is None:
            raise KeyError(f"Region with code {key} not found")

        return self._regions[index]

    def _get_by_name(self, key: str) -> Region:
        if len(key) == 0:
            raise KeyError("Empty region name")

        orig_key = key
        key = key.lower()
        index = None

        if key in self._cache:
            index = self._cache[key]
        elif key in self._index_by_name:
            index = self._index_by_name[key]
        elif key in self._index_by_short_name:
            index = self._index_by_short_name[key]
        else:
            for t in self.TYPES:
                key = key.replace(t, "")
            key.strip()

            match, score = process.extractOne(
                key, list(self._index_by_short_name))

            if score > self.THRES:
                index = self._index_by_short_name[match]

        if index is not None:
            return self._regions[index]

        raise KeyError(f"Region with name {orig_key} not found")

    def get(
        self, key: str, default: Any = Region(np.nan, np.nan, np.nan, np.nan)
    ) -> Union[Region, Any]:
        try:
            value = self[key]
            return value
        except KeyError:
            return default
