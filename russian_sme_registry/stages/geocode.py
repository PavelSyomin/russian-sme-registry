import pathlib
import shutil
import tempfile
from typing import Union

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F

from ..stages.spark_stage import SparkStage
from ..assets import get_asset_path
from ..utils.regions import Regions
from ..utils.spark_schemas import sme_geocoded_schema


def _join_name_and_type(n: Union[str, float], t: Union[str, float]) -> str:
    if pd.isna(n):
        return np.nan

    if pd.isna(t):
        return n

    prepend_types = (
        "Город", "Республика", "Поселок", "Поселок городского типа", "Рабочий поселок"
    )
    prepend = t in prepend_types
    if prepend:
        return f"{t} {n}".strip()

    return f"{n} {t}".strip()


def _join_area_and_type(a: Union[str, float], t: Union[str, float]) -> str:
    if pd.isna(a) or pd.isna(t):
        return np.nan

    if t == "г":
        return f"Город {a}"
    elif t == "р-н":
        return f"{a} район"
    elif t == "у":
        return f"{a} улус"
    else:
        return a


def _preprocess_text_column(c: pd.Series) -> pd.Series:
    if c.notna().sum() == 0:
        return c

    return c.str.upper().str.replace("Ё", "Е")


class Geocoder(SparkStage):
    ABBR_PATH = get_asset_path("abbr.csv")
    CITIES_BASE_PATH = get_asset_path("cities.csv")
    CITIES_ADDITIONAL_PATH = get_asset_path("cities_additional.csv")
    REGIONS_PATH = get_asset_path("regions.csv")
    SETTLEMENTS_PATH = get_asset_path("settlements.csv")

    ADDR_COLS = [
        "region_name",
        "region_type",
        "district_name",
        "district_type",
        "city_name",
        "city_type",
        "settlement_name",
        "settlement_type",
    ]

    DEDUPLICATION_INDEX = [
        "kind",
        "category",
        "tin",
        "reg_number",
        "first_name",
        "last_name",
        "patronymic",
        "org_name",
        "org_short_name",
        "activity_code_main",
        "activity_codes_additional",
        "region",
        "region_code",
        "region_iso_code",
        "area",
        "settlement",
        "settlement_type",
        "oktmo",
        "lat",
        "lon",
    ]

    PRODUCT_COLS = [
        "tin",
        "reg_number",
        "kind",
        "category",
        "first_name",
        "last_name",
        "patronymic",
        "org_name",
        "org_short_name",
        "activity_code_main",
        "activity_codes_additional",
        "region_iso_code",
        "region_code",
        "region",
        "area",
        "settlement",
        "settlement_type",
        "oktmo",
        "lat",
        "lon",
        "start_date",
        "end_date",
    ]

    SIGNATURE_COLS = [
        "tin",
        "reg_number",
        "kind",
        "category",
        "first_name",
        "last_name",
        "patronymic",
        "org_name",
        "org_short_name",
        "activity_code_main",
        "activity_codes_additional",
    ]

    CHUNKSIZE = 1e6

    SPARK_APP_NAME = "Geocoder"

    def __init__(self):
        super().__init__(start_spark=False)

        self._mode = None

        self._abbr = None
        self._cities = None
        self._regions = None
        self._settlements = None

        self._cities_lookup = None
        self._settlements_lookup = None
        self._joint_geodata = None

        self._setup_geodata()

    def __call__(self, in_file: str, out_file: str):
        if not pathlib.Path(in_file).exists():
            print(f"Input file {in_file} not found")
            return

        data = pd.read_csv(in_file, dtype=str, chunksize=self.CHUNKSIZE)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_df_file = str(pathlib.Path(temp_dir) / "temp_df.csv")
            for i, chunk in enumerate(data):
                print(
                    f"Processing chunk #{i}:"
                    f" {i * self.CHUNKSIZE:.0f}–{(i + 1) * self.CHUNKSIZE:.0f}"
                )

                self._detect_mode(chunk)

                if not self._check_structure(chunk):
                    print("Input data is not suitable for geocoding")
                    return

                chunk = self._fix_structure(chunk)

                addresses = self._get_addresses(chunk)

                mapping = self._geocode(addresses)
                chunk = self._remove_raw_addresses(chunk)

                chunk = chunk.merge(mapping, how="left")

                initial_count = len(chunk)
                chunk = chunk.merge(self._joint_geodata, how="left", on=["geo_id", "type"])
                assert len(chunk) == initial_count
                chunk.drop(columns=["geo_id", "type"], inplace=True)

                initial_count = len(chunk)
                chunk = chunk.merge(
                    addresses[["id", "region"]],
                    how="left",
                    on="id")
                assert len(chunk) == initial_count
                chunk["region"] = chunk["region_x"].combine_first(chunk["region_y"])
                chunk.drop(
                    columns=["region_x", "region_y", "region_code"],
                    inplace=True,
                    errors="ignore",
                )

                chunk = self._normalize_region_names(chunk)
                chunk = self._process_federal_cities(chunk)

                self._save(chunk, temp_df_file)

            if self._mode == "chain":
                self._init_spark()
                deduplicated = self._remove_duplicates(temp_df_file)
                self._write(deduplicated, out_file)
            else:
                shutil.move(temp_df_file, out_file)

    def _check_structure(self, data: pd.DataFrame) -> bool:
        missing_columns = [
            col for col in self.ADDR_COLS
            if col not in data.columns and "_type" not in col
        ]
        if missing_columns:
            print(
                f"Column(s) {', '.join(missing_columns)} are/is required"
                " but not found in data"
            )
            return False

        return True

    def _fix_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        # Ensure ids are present and correct (they are used to join mapped addresses)
        if "id" not in data.columns:
            data["id"] = range(0, data.shape[0])
        elif data["id"].nunique() != len(data):
            print(f"Found duplicates in id column, fixing")
            data["id"] = range(0, data.shape[0])

        # Add missing non-mandatory address elements
        for col in self.ADDR_COLS:
            if col not in data.columns and "_type" in col:
                data[col] = np.nan

        return data

    def _detect_mode(self, data: pd.DataFrame):
        if self._mode is not None: # already detected
            return

        if any(col not in data.columns for col in self.SIGNATURE_COLS):
            print(
                "Seems that you are running the geocoder in standalone mode"
                " (that is, to process the data other than generated by this app"
                " at the aggregate stage). This is OK, and the dataset will be geocoded"
                " as usual, but deduplication and selection of columns won't work"
            )
            self._mode = "standalone"
        else:
            self._mode = "chain"

    def _get_addresses(self, data: pd.DataFrame) -> pd.DataFrame:
        addresses = data.loc[:, ["id"] + self.ADDR_COLS]
        addresses = self._normalize_address_elements_types(addresses)
        addresses = self._normalize_region_names(addresses)
        addresses.iloc[:, 1:] = addresses.iloc[:, 1:].apply(_preprocess_text_column)

        return addresses

    def _build_cities_lookup(self) -> pd.DataFrame:
        std = self._cities[["id", "region", "area", "city", "settlement"]]
        cities_from_areas = self._cities.loc[(self._cities["area_type"] == "г") & (self._cities["city"].isna())].copy()
        cities_from_areas["city"] = cities_from_areas["area"]
        cities_from_areas["area"] = np.nan
        cities_from_areas = cities_from_areas[["id", "region", "area", "city", "settlement"]]
        std = pd.concat((std, cities_from_areas))

        std = self._normalize_region_names(std)
        std.iloc[:, 1:] = std.iloc[:, 1:].apply(_preprocess_text_column)

        return std

    def _build_settlements_lookup(self) -> pd.DataFrame:
        std = self._settlements.loc[:, ["id", "region", "municipality", "settlement", "type"]]
        std = self._expand_abbrs(std, "type")
        std = self._normalize_region_names(std)
        std.iloc[:, 1:] = std.iloc[:, 1:].apply(_preprocess_text_column)

        return std

    def _normalize_address_elements_types(
        self, addresses: pd.DataFrame
    ) -> pd.DataFrame:
        for option in ("region", "district", "city", "settlement"):
            target_col = f"{option}_type"
            addresses = self._expand_abbrs(addresses, target_col)

            parts = [f"{option}_name", f"{option}_type"]
            addresses[option] = addresses[parts].apply(
                lambda row: _join_name_and_type(row[parts[0]], row[parts[1]]),
                axis=1
            )

        return addresses

    def _expand_abbrs(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        initial_count = len(data)
        data[column] = data[column].fillna("").str.upper()
        data = data.merge(
            self._abbr,
            how="left",
            left_on=column,
            right_on="name",
        )
        data[column] = data["name_full"]
        data.drop(columns=self._abbr.columns, inplace=True)
        assert len(data) == initial_count, (
            f"Number of items must not change, but for {column} "
            f"the size has changed: {initial_count} -> {len(data)}"
        )

        return data

    def _normalize_region_names(self, data: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(data)
        regions = data["region"].dropna().unique()

        regions_norm = []
        regions_codes = []
        regions_iso_codes = []
        for region in regions:
            region_info = self._regions.get(region)
            regions_norm.append(region_info.name)
            regions_codes.append(region_info.code)
            regions_iso_codes.append(region_info.iso_code)

        regions_norm = pd.DataFrame({
            "region": regions,
            "region_norm": regions_norm,
            "region_code": regions_codes,
            "region_iso_code": regions_iso_codes,
        })

        data = data.merge(regions_norm, how="left", on="region")
        data["region"] = data["region_norm"]
        data.drop(columns="region_norm", inplace=True)
        assert len(data) == initial_count

        return data

    def _setup_geodata(self):
        print("Loading geodata")

        self._load_abbr()
        self._load_cities()
        self._load_regions()
        self._load_settlements()

        self._build_lookup_tables()

    def _load_abbr(self):
        abbr = pd.read_csv(self.ABBR_PATH)

        short_to_full = abbr[["name", "name_full"]]

        full_to_full = abbr[["name_full", "name_full"]]
        full_to_full.columns = ("name", "name_full")

        without_dots = abbr.loc[~abbr["name"].str.endswith("."), ["name", "name_full"]]
        without_dots["name"] = without_dots["name"] + "."

        abbr_to_full = pd.concat((
            short_to_full,
            full_to_full,
            without_dots
        ))
        abbr_to_full = abbr_to_full.apply(lambda x: x.str.upper())
        abbr_to_full.drop_duplicates("name", inplace=True)

        self._abbr = abbr_to_full
        print("Loaded address abbreviations")

    def _load_cities(self):
        cities_base = pd.read_csv(self.CITIES_BASE_PATH, dtype=str)
        cities_additional = pd.read_csv(self.CITIES_ADDITIONAL_PATH, dtype=str)
        cities = pd.concat((cities_base, cities_additional))
        cities.reset_index(drop=True, inplace=True)
        cities["id"] = range(0, cities.shape[0])
        self._cities = cities
        print("Loaded cities")

    def _load_regions(self):
        self._regions = Regions(self.REGIONS_PATH)
        print("Loaded regions")

    def _load_settlements(self):
        self._settlements = pd.read_csv(self.SETTLEMENTS_PATH, dtype=str)
        print("Loaded settlements")

    def _build_lookup_tables(self):
        self._cities_lookup = self._build_cities_lookup()
        self._settlements_lookup = self._build_settlements_lookup()
        self._joint_geodata = self._build_joint_geodata()
        print("Builded lookup tables")

    def _geocode(
        self,
        addresses: pd.DataFrame,
    ) -> pd.DataFrame:
        merge_options = [
            {
                "name": "Settlements by all parts with full district name",
                "addresses": ["region", "district", "settlement_name", "settlement_type"],
                "standard": ["region", "municipality", "settlement", "type"],
                "type": "settlements"
            },
            {
                "name": "Settlements by all parts with partial district name (no type)",
                "addresses": ["region", "district_name", "settlement_name", "settlement_type"],
                "standard": ["region", "municipality", "settlement", "type"],
                "type": "settlements",
            },
            {
                "name": "Settlements by all parts with full city name",
                "addresses": ["region", "city", "settlement_name", "settlement_type"],
                "standard": ["region", "municipality", "settlement", "type"],
                "type": "settlements",
            },
            {
                "name": "Settlements by all parts with partial city name (no type)",
                "addresses": ["region", "city_name", "settlement_name", "settlement_type"],
                "standard": ["region", "municipality", "settlement", "type"],
                "type": "settlements",
            },
            {
                "name": "Settlements by all parts except for type with full district name",
                "addresses": ["region", "district", "settlement_name"],
                "standard": ["region", "municipality", "settlement"],
                "type": "settlements",
            },
            {
                "name": "Settlements by all parts except for type with partial district name",
                "addresses": ["region", "district_name", "settlement_name"],
                "standard": ["region", "municipality", "settlement"],
                "type": "settlements",
            },
            {
                "name": "Settlements by all parts except for type with full city name",
                "addresses": ["region", "city", "settlement_name"],
                "standard": ["region", "municipality", "settlement"],
                "type": "settlements",
            },
            {
                "name": "Settlements by all parts except for type with partial city name",
                "addresses": ["region", "city_name", "settlement_name"],
                "standard": ["region", "municipality", "settlement"],
                "type": "settlements",
            },
            {
                "name": "Settlements by region and settlement with type",
                "addresses": ["region", "settlement_name", "settlement_type"],
                "standard": ["region", "settlement", "type"],
                "type": "settlements",
            },
            {
                "name": "Settlements by region and settlement without type",
                "addresses": ["region", "settlement_name"],
                "standard": ["region", "settlement"],
                "type": "settlements",
            },
            {
                "name": "Cities by all parts",
                "addresses": ["region", "district_name", "city_name", "settlement_name"],
                "standard": ["region", "area", "city", "settlement"],
                "type": "cities",
            },
            {
                "name": "Cities by all parts except for settlements",
                "addresses": ["region", "district_name", "city_name"],
                "standard": ["region", "area", "city"],
                "type": "cities",
            },
            {
                "name": "Cities by region and city",
                "addresses": ["region", "city_name"],
                "standard": ["region", "city"],
                "type": "cities",
            },
            {
                "name": "Cities by region and district-as-city",
                "addresses": ["region", "city_name"],
                "standard": ["region", "area"],
                "type": "cities",
            },
        ]

        mappings = []
        rest = addresses
        orig_cols = addresses.columns
        for option in merge_options:
            name = option["name"]
            left_cols = option["addresses"]
            right_cols = option["standard"]
            type_ = option["type"]

            to_merge = rest[orig_cols]
            standard = self._cities_lookup.copy() if type_ == "cities" else self._settlements_lookup.copy()
            standard.drop_duplicates(subset=right_cols, keep=False, inplace=True)
            if len(right_cols) == 2:
                standard.dropna(subset=right_cols, inplace=True)
            standard.rename(columns={"id": "geo_id"}, inplace=True)

            size_before = len(to_merge)
            merged = to_merge.merge(
                standard,
                how="left",
                left_on=left_cols,
                right_on=right_cols,
                suffixes=("", "_x")
            )

            size_after = len(merged)
            assert size_before == size_after

            mapped = merged.loc[merged["geo_id"].notna(), ["id", "geo_id"]]
            mapped["type"] = type_[0]
            if len(mapped) > 0:
                mappings.append(mapped)

            rest = merged.loc[merged["geo_id"].isna()]

            print(f"Option {name}: found {len(mapped)} matches, {len(rest)} records left")

        addr_to_geo = pd.concat(mappings)

        return addr_to_geo

    def _build_joint_geodata(self) -> pd.DataFrame:
        s_cols = [
            "id", "region", "municipality", "settlement", "type",
            "oktmo", "longitude_dd", "latitude_dd"
        ]
        s = self._settlements[s_cols].copy()
        s["geosource_type"] = "s"
        s.rename(columns={
            "id": "geo_id",
            "municipality": "area",
            "type": "settlement_type",
            "longitude_dd": "lon",
            "latitude_dd": "lat",
            "geosource_type": "type",
        }, inplace=True)

        c_cols = [
            "id", "region", "area", "area_type", "city", "city_type",
            "settlement", "settlement_type", "oktmo", "geo_lat", "geo_lon"
        ]
        c = self._cities[c_cols].copy()
        c["settlement"] = (
            c["settlement"]
            .combine_first(c["city"])
            .combine_first(c["area"])
            .reset_index(drop=True)
        )
        c.loc[c["area_type"] == "г", "area"] = np.nan
        c["area"] = c[["area", "area_type"]].apply(
            lambda x: _join_area_and_type(x.iloc[0], x.iloc[1]), axis=1)

        c["geosource_type"] = "c"
        c["settlement_type"] = "г"
        c.rename(columns={
            "id": "geo_id",
            "geo_lat": "lat",
            "geo_lon": "lon",
            "geosource_type": "type",
        }, inplace=True)
        c.drop(columns=["area_type", "city", "city_type"], inplace=True)

        geodata = pd.concat((c, s))

        return geodata

    def _remove_raw_addresses(self, data: pd.DataFrame) -> pd.DataFrame:
        data.drop(columns=self.ADDR_COLS, inplace=True)

        return data

    def _remove_duplicates(self, in_file: str) -> DataFrame:
        print("Removing duplicates that may have appeared after geocoding")

        w_for_row_number = (
            Window
            .partitionBy(self.DEDUPLICATION_INDEX)
            .orderBy("start_date")
        )
        w_for_end_date = w_for_row_number.rowsBetween(0, Window.unboundedFollowing)
        w_by_tin = Window.partitionBy(["tin"]).orderBy("start_date")
        w_by_tin_unbounded = w_by_tin.rowsBetween(0, Window.unboundedFollowing)

        data = self._read(in_file, sme_geocoded_schema)

        initial_count = data.count()
        deduplicated = (
            data
            .withColumn("hash", F.hash(*self.DEDUPLICATION_INDEX))
            .withColumn("prev_hash", F.lag("hash", default=0).over(w_by_tin))
            .withColumn("hash_change", F.col("hash") != F.col("prev_hash"))
            .withColumn("sme_entity_end_date", F.last("end_date").over(w_by_tin_unbounded))
            .filter("hash_change = true")
            .withColumn("end_date", F.lead("start_date").over(w_by_tin))
            .withColumn("end_date", F.coalesce("end_date", "sme_entity_end_date"))
            .drop("hash", "prev_hash", "hash_change", "sme_entity_end_date")
            .orderBy("tin", "start_date")
        )
        after_count = deduplicated.count()

        print(f"Removed {initial_count - after_count} duplicated rows")

        return deduplicated

    def _process_federal_cities(self, data: pd.DataFrame) -> pd.DataFrame:
        for city in ("Москва", "Санкт-Петербург"):
            data.loc[
                (data["region"] == city) & data["settlement"].isna(),
                "settlement"
            ] = city

        return data

    def _save(self, data: pd.DataFrame, out_file: str):
        product = data[self.PRODUCT_COLS] if self._mode == "chain" else data

        out_file = pathlib.Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_file.exists():
            product.to_csv(out_file, index=False, header=False, mode="a")
        else:
            product.to_csv(out_file, index=False)
