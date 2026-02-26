from typing import List, Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import broadcast
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, StructField, StructType

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

        Implements gaps-and-islands logic to merge duplicate rows (same attributes
        across consecutive dates) into single rows with start_date and end_date.
        A gap is detected when a date exists in the table (for other TINs) but
        not for this TIN—such gaps split islands into separate rows.
        """
        data = self._read(
            in_dir,
            sme_schema,
            dateFormat=self.INPUT_DATE_FORMAT,
            add_input_file=True,
        )
        if data is None:
            return

        data = self._validate_and_fix_data_dates_per_file(
            data,
            file_col="_input_file",
            date_col="data_date",
        )

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

        table = self._deduplicate_gaps_and_islands(
            data_prepared,
            date_col="data_date",
            id_col="tin",
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

    def _validate_and_fix_data_dates_per_file(
        self,
        data: DataFrame,
        *,
        file_col: str,
        date_col: str,
        low_freq_threshold: float = 0.01,
    ) -> DataFrame:
        """Validate data_date per input file: each file should have one date.
        If multiple dates exist and a date has frequency < threshold, replace
        it with the majority date. Otherwise raise RuntimeError.
        """
        file_date_counts = data.groupBy(file_col, date_col).count()
        file_totals = data.groupBy(file_col).count().withColumnRenamed(
            "count", "_file_total"
        )
        file_stats = (
            file_date_counts.join(file_totals, file_col)
            .withColumn("_freq", F.col("count") / F.col("_file_total"))
        )

        w = Window.partitionBy(file_col).orderBy(F.desc("count"))
        majority = (
            file_stats.withColumn("_rank", F.row_number().over(w))
            .filter("_rank == 1")
            .select(
                F.col(file_col).alias("_file"),
                F.col(date_col).alias("_majority_date"),
            )
        )

        correctable = (
            file_stats.filter(F.col("_freq") < low_freq_threshold)
            .join(majority, F.col(file_col) == F.col("_file"))
            .select(
                F.col(file_col).alias("_fc_file"),
                F.col(date_col).alias("_wrong_date"),
                F.col("_majority_date"),
            )
        )

        high_freq_dates = file_stats.filter(F.col("_freq") >= low_freq_threshold)
        files_with_multiple = (
            file_stats.groupBy(file_col)
            .agg(F.count("*").alias("_num_dates"))
            .filter("_num_dates > 1")
        )
        problematic = (
            files_with_multiple.join(high_freq_dates, file_col)
            .join(majority, F.col(file_col) == F.col("_file"))
            .filter(F.col(date_col) != F.col("_majority_date"))
        )
        problematic_files = problematic.select(file_col).distinct()
        problematic_count = problematic_files.count()
        if problematic_count > 0:
            bad_files = problematic_files.limit(10).collect()
            raise RuntimeError(
                f"Data quality error: {problematic_count} file(s) have multiple "
                f"data_date values with frequency >= {low_freq_threshold:.0%}. "
                f"Each input file must have exactly one date. "
                f"Affected files (sample): {[r[0] for r in bad_files]}"
            )

        if correctable.count() > 0:
            data = (
                data.join(
                    correctable,
                    (F.col(file_col) == F.col("_fc_file"))
                    & (F.col(date_col) == F.col("_wrong_date")),
                    "left",
                )
                .withColumn(
                    date_col,
                    F.coalesce(F.col("_majority_date"), F.col(date_col)),
                )
                .drop("_fc_file", "_wrong_date", "_majority_date")
            )
            print(
                f"Corrected low-frequency data_date values "
                f"(<{low_freq_threshold:.0%}) to majority date per file"
            )

        return data.drop(file_col)

    def _warn_low_frequency_calendar_dates(
        self,
        data: DataFrame,
        *,
        date_col: str,
        low_freq_threshold: float = 0.001,
    ) -> None:
        """Warn if any date in the calendar has extremely low frequency."""
        total = data.count()
        if total == 0:
            return
        date_counts = data.groupBy(date_col).count()
        date_counts = date_counts.withColumn(
            "_freq", F.col("count") / F.lit(total)
        )
        low_freq = date_counts.filter(F.col("_freq") < low_freq_threshold)
        low_freq_list = low_freq.collect()
        if low_freq_list:
            print(
                f"WARNING: {len(low_freq_list)} date(s) have very low frequency "
                f"(<{low_freq_threshold:.2%}) in the data. This may indicate "
                f"data quality issues and could affect gap detection:"
            )
            for row in low_freq_list[:10]:
                print(f"  {row[date_col]}: {row['count']} rows ({row['_freq']:.4%})")
            if len(low_freq_list) > 10:
                print(f"  ... and {len(low_freq_list) - 10} more")

    def _deduplicate_gaps_and_islands(
        self,
        data: DataFrame,
        *,
        date_col: str,
        id_col: str,
        hash_cols: List[str],
        output_cols: List[str],
    ) -> DataFrame:
        """Merge duplicate rows (same attributes across consecutive dates) into
        intervals with start_date and end_date. Uses gaps-and-islands logic with
        gap detection: a gap exists when a date appears in the table for other
        IDs but not for this one.
        """
        w = Window.partitionBy(id_col).orderBy(date_col)
        cols_for_group = [
            c for c in output_cols
            if c not in ("start_date", "end_date", id_col)
        ]

        self._warn_low_frequency_calendar_dates(data, date_col=date_col)

        dates_sorted = sorted(
            row[0] for row in data.select(date_col).distinct().collect()
        )
        calendar_data = [
            (d, dates_sorted[i + 1] if i + 1 < len(dates_sorted) else None)
            for i, d in enumerate(dates_sorted)
        ]
        calendar_schema = StructType([
            StructField(date_col, DateType(), False),
            StructField("next_global_date", DateType(), True),
        ])
        global_calendar = broadcast(
            self._session.createDataFrame(calendar_data, schema=calendar_schema)
        )

        data_with_cal = data.join(global_calendar, on=date_col, how="left")
        data_with_cal = data_with_cal.withColumn(
            "hash", F.hash(*hash_cols)
        )

        prev_next_global = F.lag("next_global_date").over(w)
        df_analysis = (
            data_with_cal
            .withColumn("prev_hash", F.lag("hash", default=0).over(w))
            .withColumn(
                "started_due_to_gap",
                (prev_next_global != F.col(date_col)) & F.col("prev_hash").isNotNull(),
            )
            .withColumn(
                "is_new_island",
                F.col("prev_hash").isNull()
                | (F.col("prev_hash") != F.col("hash"))
                | (prev_next_global != F.col(date_col)),
            )
        )
        df_islands = df_analysis.withColumn(
            "island_id",
            F.sum(F.when(F.col("is_new_island"), 1).otherwise(0)).over(
                w.rowsBetween(Window.unboundedPreceding, 0)
            ),
        )

        agg_exprs = [
            F.min(date_col).alias("start_date"),
            F.max(date_col).alias("max_date"),
            F.max(F.struct(date_col, "next_global_date")).getField(
                "next_global_date"
            ).alias("last_next_global"),
            F.max(F.when(F.col("started_due_to_gap"), 1).otherwise(0)).alias(
                "_started_due_to_gap"
            ),
        ]
        agg_exprs.extend([F.first(c).alias(c) for c in cols_for_group])

        grouped = df_islands.groupBy(id_col, "island_id").agg(*agg_exprs)

        w_final = Window.partitionBy(id_col).orderBy("start_date")
        result = (
            grouped
            .withColumn(
                "end_date",
                F.when(
                    F.lead("_started_due_to_gap").over(w_final) == 1,
                    F.col("last_next_global"),
                ).otherwise(F.coalesce(F.lead("start_date").over(w_final), F.col("max_date"))),
            )
            .drop("_started_due_to_gap", "last_next_global", "max_date", "island_id")
            .select(*output_cols)
            .orderBy([id_col, "start_date"])
        )

        return result

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
