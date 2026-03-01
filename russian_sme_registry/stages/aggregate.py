from typing import List, Optional

from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F

from ..stages.spark_stage import SparkStage
from ..utils.enums import SourceDatasets
from ..utils.spark_schemas import (
    sme_schema, sme_aggregated_schema, revexp_schema, empl_schema
)


class Aggregator(SparkStage):
    INPUT_DATE_FORMAT = "dd.MM.yyyy"
    SPARK_APP_NAME = "Extracted Data Aggregator"

    def __call__(
        self,
        in_dir: str,
        out_file: str,
        source_dataset: str,
        sme_data_file: Optional[str] = None,
        with_crimea: bool = False,
        with_new_territories: bool = False,
    ):
        """Execute the aggregation of all datasets"""
        if source_dataset == SourceDatasets.sme.value:
            self._process_sme_registry(in_dir, out_file, with_crimea, with_new_territories)
        elif source_dataset == SourceDatasets.revexp.value:
            self._process_revexp_data(in_dir, out_file, sme_data_file)
        elif source_dataset == SourceDatasets.empl.value:
            self._process_empl_data(in_dir, out_file, sme_data_file)
        else:
            raise RuntimeError(
                f"Unsupported source dataset {source_dataset}, "
                f"expected one of {[sd.value for sd in SourceDatasets]}"
            )

    def _filter_by_tins(self, table: DataFrame, sme_data_file: str) -> DataFrame:
        print("Filtering by TINs")

        sme_data = self._read(sme_data_file, sme_aggregated_schema)

        tins = sme_data.filter("kind == 1").select("tin")

        table = table.join(tins, on="tin", how="leftsemi")

        return table

    def _process_sme_registry(
        self,
        in_dir: str,
        out_file: str,
        with_crimea: bool = False,
        with_new_territories: bool = False,
    ):
        """Process CSV files extracted from SME registry archives.

        Islands are defined by (tin, reestr_date): exclusion and re-entry updates
        reestr_date. Within each island, merge duplicate rows (same attributes
        across consecutive data_date) into intervals. start_date = first data_date
        of each group, end_date = last data_date. Empty reestr_date falls back
        to data_date for island partitioning.
        """
        data = self._read(
            in_dir,
            sme_schema,
            dateFormat=self.INPUT_DATE_FORMAT,
        )
        if data is None:
            return

        cols_to_check_for_duplicates = [
            "kind", "category", "tin", "reg_number",
            "first_name", "last_name", "patronymic",
            "org_name", "org_short_name",
            "region_name",
            "district_name", "city_name", "settlement_name",
            "activity_code_main",
        ]
        cols_to_select = [
            "kind",
            "category",
            "tin",
            "reg_number",
            "first_name",
            "last_name",
            "patronymic",
            "org_name",
            "org_short_name",
            "region_code",
            "region_name",
            "region_type",
            "district_name",
            "district_type",
            "city_name",
            "city_type",
            "settlement_name",
            "settlement_type",
            "activity_code_main",
            "start_date",
            "end_date",
        ]
        cols_to_uppercase = [
            "first_name", "last_name", "patronymic",
            "org_name", "org_short_name",
            "region_name", "region_type",
            "district_name", "district_type",
            "city_name", "city_type",
            "settlement_name", "settlement_type",
        ]

        w_by_tin = Window.partitionBy(["tin"]).orderBy("data_date")
        w_by_tin_unbounded = w_by_tin.rowsBetween(0, Window.unboundedFollowing)

        excluded_regions = []
        if not with_crimea:
            excluded_regions.extend([
                "Крым",
                "Севастополь",
            ])
        if not with_new_territories:
            excluded_regions.extend([
                "Донецкая",
                "Луганская",
                "Запорожская",
                "Херсонская",
            ])
        if excluded_regions:
            excluded_regions_condition = (
                "not ("
                + " or ".join(f"region_name ilike '%{region.upper()}%'" for region in excluded_regions)
                + ")"
            )
            data = data.filter(excluded_regions_condition)

        data_prepared = (
            data
            .withColumns({
                colname: F.upper(colname)
                for colname in cols_to_uppercase
            })
            .withColumns({
                "ind_tin": F.lpad("ind_tin", 12, "0"),
                "org_tin": F.lpad("org_tin", 10, "0"),
            })
            .withColumns({
                "tin": F.coalesce("ind_tin", "org_tin"),
                "reg_number": F.coalesce("ind_number", "org_number"),
            })
            .withColumn("reg_number", F.first("reg_number", ignorenulls=True).over(w_by_tin_unbounded))
        )

        table = self._deduplicate_by_reestr_date(
            data_prepared,
            id_col="tin",
            reestr_date_col="reestr_date",
            data_date_col="data_date",
            hash_cols=cols_to_check_for_duplicates,
            output_cols=cols_to_select,
        ).cache()

        count_after = table.count()
        print(f"Aggregated SME table contains {count_after} rows")

        self._write(table, out_file)

    def _process_revexp_data(self, in_dir: str, out_file: str,
                             sme_data_file: Optional[str]):
        """Combine revexp CSV files into a single file filtering by TINs"""
        data = self._read(in_dir, revexp_schema, dateFormat=self.INPUT_DATE_FORMAT)
        if data is None:
            return

        window = Window.partitionBy("tin", "data_date").orderBy(F.desc("doc_date"))

        table = (
            data
            .withColumnRenamed("org_tin", "tin")
            .withColumn("tin", F.lpad("tin", 10, "0"))
        )

        if sme_data_file is not None:
            table = self._filter_by_tins(table, sme_data_file)

        table = (
            table
            .withColumn("row_number", F.row_number().over(window))
            .filter("row_number == 1")
            .select("tin", F.year("data_date").alias("year"), "revenue", "expenditure")
            .orderBy("tin", "year")
            .cache()
        )

        print(f"Revexp resulting table contains {table.count()} rows")

        self._write(table, out_file)

    def _deduplicate_by_reestr_date(
        self,
        data: DataFrame,
        *,
        id_col: str,
        reestr_date_col: str,
        data_date_col: str,
        hash_cols: List[str],
        output_cols: List[str],
    ) -> DataFrame:
        """Merge duplicate rows within islands defined by (tin, reestr_date).

        Islands: same reestr_date = same registry period. Empty reestr_date
        falls back to data_date. Within each island, group by attribute hash;
        start_date = first data_date of group, end_date = last data_date.
        Single window + single groupBy for efficiency.
        """
        island_date = F.coalesce(F.col(reestr_date_col), F.col(data_date_col))
        data = data.withColumn("_island_date", island_date).withColumn(
            "_hash", F.hash(*hash_cols)
        )

        w = Window.partitionBy(id_col, "_island_date").orderBy(data_date_col)
        prev_hash = F.lag("_hash", 1).over(w)
        is_new_group = prev_hash.isNull() | (prev_hash != F.col("_hash"))
        group_id = F.sum(F.when(is_new_group, 1).otherwise(0)).over(
            w.rowsBetween(Window.unboundedPreceding, 0)
        )

        data = data.withColumn("_group_id", group_id)

        cols_for_agg = [
            c for c in output_cols
            if c not in ("start_date", "end_date", id_col)
        ]
        agg_exprs = [
            F.min(data_date_col).alias("start_date"),
            F.max(data_date_col).alias("end_date"),
        ]
        agg_exprs.extend([F.first(c).alias(c) for c in cols_for_agg])

        grouped = data.groupBy(id_col, "_island_date", "_group_id").agg(*agg_exprs)

        return (
            grouped.drop("_island_date", "_group_id")
            .select(*output_cols)
            .orderBy([id_col, "start_date"])
        )

    def _process_empl_data(self, in_dir: str, out_file: str,
                           sme_data_file: Optional[str]):
        """Combine employees CSV files into a single file filtering by TINs"""
        data = self._read(in_dir, empl_schema, dateFormat=self.INPUT_DATE_FORMAT)
        if data is None:
            return

        window = Window.partitionBy("tin", "data_date").orderBy(F.desc("doc_date"))

        table = (
            data
            .withColumnRenamed("org_tin", "tin")
            .withColumn("tin", F.lpad("tin", 10, "0"))
        )

        if sme_data_file is not None:
            table = self._filter_by_tins(table, sme_data_file)

        table = (
            table
            .withColumn("row_number", F.row_number().over(window))
            .filter("row_number = 1")
            .select("tin", F.year("data_date").alias("year"), "employees_count")
            .orderBy("tin", "year")
            .cache()
        )

        print(f"Revexp resulting table contains {table.count()} rows")

        self._write(table, out_file)
