from typing import Optional

from pyspark.sql.types import ByteType
import pyspark.sql.functions as F

from ..stages.spark_stage import SparkStage
from ..utils.spark_schemas import (
    sme_geocoded_schema, revexp_agg_schema, empl_agg_schema
)


class Panelizer(SparkStage):
    SPARK_APP_NAME = "Panel Table Maker"

    def __call__(
            self,
            sme_file: str,
            out_file: str,
            revexp_file: Optional[str] = None,
            empl_file: Optional[str] = None,
            remove_personal_names: bool = True
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
                "activity_codes_additional": "additional_nace_codes",
                "region": "region_name",
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

        self._write(panel, out_file, nullValue="NA", sep=";")
