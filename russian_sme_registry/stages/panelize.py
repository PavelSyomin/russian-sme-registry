from typing import Optional

import pyspark.sql.functions as F

from ..stages.spark_stage import SparkStage
from ..utils.spark_schemas import (
    sme_geocoded_schema, revexp_agg_schema, empl_agg_schema
)


class Panelizer(SparkStage):
    SPARK_APP_NAME = "Panel Table Maker"

    def __call__(self, sme_file: str, out_file: str,
                 revexp_file: Optional[str] = None,
                 empl_file: Optional[str] = None):
        sme_data = self._read(sme_file, sme_geocoded_schema)
        if sme_data is None:
            return

        panel = (
            sme_data
            .withColumn(
                "year",
                F.explode(F.sequence(F.year("start_date"), F.year("end_date")))
            )
        )

        if revexp_file is not None:
            revexp_data = self._read(revexp_file, revexp_agg_schema)
            if revexp_data is not None:
                print("Joining with revexp data")
                panel = panel.join(revexp_data, on=["tin", "year"], how="leftouter")

        if empl_file is not None:
            empl_data = self._read(empl_file, empl_agg_schema)
            if empl_data is not None:
                print("Joining with empl data")
                panel = panel.join(empl_data, on=["tin", "year"], how="leftouter")

        panel = panel.orderBy("tin", "year")

        self._write(panel, out_file)
