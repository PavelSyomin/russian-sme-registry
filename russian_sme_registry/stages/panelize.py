import pathlib
from typing import Optional, Dict

from pyspark.sql.types import ByteType
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window

from ..stages.spark_stage import SparkStage
from ..utils.spark_schemas import (
    sme_geocoded_schema, revexp_agg_schema, empl_agg_schema
)


class Panelizer(SparkStage):
    EXCEL_ROWS_LIMIT = 1e6
    SPARK_APP_NAME = "Panel Table Maker"

    def __call__(
            self,
            sme_file: str,
            out_dir: str,
            revexp_file: Optional[str] = None,
            empl_file: Optional[str] = None,
            remove_personal_names: bool = True,
            save_to_csv: bool = True,
            save_to_parquet: bool = False,
            save_to_excel: bool = False,
            split_by_region: bool = False,
        ):
        sme_data = self._read(sme_file, sme_geocoded_schema)
        if sme_data is None:
            return

        panel = (
            sme_data
            .withColumn(
                "year",
                F.explode(F.sequence(F.year("start_date"), F.year("end_date")))
            )
            .withColumn(
                "kind", (F.col("kind") == 2).cast(ByteType())
            )
            .withColumnsRenamed({
                "tin": "tax_number",
                "reg_number": "registration_number",
                "kind": "is_sole_trader",
                "category": "sme_category",
                "activity_code_main": "main_nace_code",
                "oktmo": "municipality_code",
            })
        )

        if remove_personal_names:
            panel = panel.drop("first_name", "last_name", "patronymic")

        if revexp_file is not None:
            revexp_data = self._read(revexp_file, revexp_agg_schema)
            if revexp_data is not None:
                print("Joining with revexp data")
                revexp_data = revexp_data.withColumnsRenamed({
                    "tin": "tax_number",
                })
                panel = panel.join(revexp_data, on=["tax_number", "year"], how="leftouter")

        if empl_file is not None:
            empl_data = self._read(empl_file, empl_agg_schema)
            if empl_data is not None:
                print("Joining with empl data")
                empl_data = empl_data.withColumnsRenamed({
                    "tin": "tax_number",
                })
                panel = panel.join(empl_data, on=["tax_number", "year"], how="leftouter")

        panel = panel.orderBy("tax_number", "year")

        if split_by_region:
            parts = self._split_by_region(panel)
        else:
            parts = {"all-regions": panel}

        if save_to_csv:
            self._save_csv(parts, out_dir)
        if save_to_parquet:
            self._save_parquet(parts, out_dir)
        if save_to_excel:
            self._save_excel(parts, out_dir)

    def _split_by_region(self, panel: DataFrame) -> Dict[str, DataFrame]:
        regions = [
            row.region for
            row in panel.select("region").distinct().sort("region").collect()
        ]

        parts = {}
        for region in regions:
            part = panel.filter(F.col("region") == region)
            if str(region) == "NaN":
                region = "unknown-regions"
            parts[region] = part

        return parts

    def _save_csv(self, parts: Dict[str, DataFrame], out_dir: str):
        path = pathlib.Path(out_dir) / "csv"
        path.mkdir(parents=True, exist_ok=True)

        for region, part in parts.items():
            out_file = path / f"{region}.csv"
            self._write(part, out_file, nullValue="NA", sep=";")

    def _save_parquet(self, parts: Dict[str, DataFrame], out_dir: str):
        path = pathlib.Path(out_dir) / "parquet"
        path.mkdir(parents=True, exist_ok=True)

        for region, part in parts.items():
            out_file = path / f"{region}.parquet"
            self._write(part, out_file, format="parquet")

    def _save_excel(self, parts: Dict[str, DataFrame], out_dir: str):
        path = pathlib.Path(out_dir) / "excel"
        path.mkdir(parents=True, exist_ok=True)

        for region, part in parts.items():
            n_chunks = (part.count() // self.EXCEL_ROWS_LIMIT) + 1
            if n_chunks == 1:
                out_file = path / f"{region}.xlsx"
                part.toPandas().to_excel(out_file)
                print(f"Saved to {out_file}")
            else:
                part = part.withColumn(
                    "chunk",
                    F.row_number().over(Window.partitionBy().orderBy("year")) // self.EXCEL_ROWS_LIMIT
                )
                for i in range(n_chunks):
                    out_file = path / f"{region} (part {i+1} of {n_chunks}).xlsx"
                    part.filter(F.col("chunk") == i).drop("chunk").toPandas().to_excel(out_file)
                    print(f"Saved to {out_file}")
