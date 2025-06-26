import pathlib
import shutil
import tempfile

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType
from ..utils.helpers import require_java


class SparkStage:
    SPARK_APP_NAME = "Generic Spark Stage"

    def __init__(self, start_spark: bool = True):
        self._session = None
        require_java()  # Check for Java before any Spark operations

        if start_spark:
            self._init_spark()

    def __del__(self):
        if self._session is not None:
            print("Stopping Spark")
            self._session.stop()

    def _init_spark(self):
        """Spark configuration and initialization"""
        print("Starting Spark")
        self._session = (
            SparkSession
            .builder
            .master("local")
            .appName(self.SPARK_APP_NAME)
            .getOrCreate()
        )

        web_url = self._session.sparkContext.uiWebUrl
        print(f"Spark session has started. You can monitor it at {web_url}")

    def _read(self, in_path: str, schema: StructType, **kwargs) -> DataFrame:
        path = pathlib.Path(in_path)
        if not path.exists():
            print(f"Input path {in_path} not found")
            return None

        if path.is_dir():
            input_files = [str(fn.absolute()) for fn in path.glob("data-*.csv")]
        elif path.suffix in (".csv",):
            input_files = [str(path.absolute())]
        else:
            input_files = []

        if len(input_files) == 0:
            print("Input path does not contain readable CSV file(s)")
            return None

        options = {
            "header": True,
            "escape": '"',
        }
        options.update(kwargs)

        print(f"Reading source data at {in_path}")

        data = self._session.read.options(**options).schema(schema).csv(input_files)

        print(f"Source CSV(s) contain(s) {data.count()} rows")

        return data

    def _write(self, df: DataFrame, out_file: str, format: str = "csv", **kwargs):
        """Save Spark dataframe into a single file"""
        if format in ("csv",):
            options = dict(header=True, escape='"')
        elif format in ("parquet",):
            options = dict()
        else:
            raise ValueError(f"Unsupported format: {format}")

        options.update(**kwargs)

        with tempfile.TemporaryDirectory() as out_dir:
            df.coalesce(1).write.options(**options).format(format).save(out_dir, mode="overwrite")

            # Spark writes to a folder with an arbitrary filename,
            # so we need to find and move the resulting file to the destination
            result = next(pathlib.Path(out_dir).glob(f"*.{format}"), None)
            if result is None:
                print("Failed to save file")

            pathlib.Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(result, out_file)

            print(f"Saved to {out_file}")
