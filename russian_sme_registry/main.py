import json
import os
import pathlib
from typing import List, Optional
from typing_extensions import Annotated

import typer

from russian_sme_registry.stages.aggregate import Aggregator
from russian_sme_registry.stages.download import Downloader
from russian_sme_registry.stages.extract import Extractor
from russian_sme_registry.stages.geocode import DaDataGeocoder, LocalGeocoder
from russian_sme_registry.stages.panelize import Panelizer
from russian_sme_registry.utils.enums import Geocoders, SourceDatasets, StageNames, Storages
from russian_sme_registry.utils.helpers import print_config


APP_NAME = "russian-sme-registry"

app = typer.Typer(
    help=(
        "A tool to create a geocoded CSV dataset of Russian small and medium-sized "
        "enterprises from Federal Tax Service (FTS) open data"
    ),
    rich_markup_mode="markdown"
)
download_app = typer.Typer()
extract_app = typer.Typer()
aggregate_app = typer.Typer()
app.add_typer(
    download_app,
    name="download",
    help="Download source zipped dataset(s) from FTS open data server (stage 1)",
    rich_help_panel="Stages",
    no_args_is_help=True
)
app.add_typer(
    extract_app,
    name="extract",
    help="Extract data from downloaded source zipped datasets to CSV files (stage 2)",
    rich_help_panel="Stages",
    no_args_is_help=True
)
app.add_typer(
    aggregate_app,
    name="aggregate",
    help="Aggregate extracted data into a single CSV file removing duplicates (stage 3)",
    rich_help_panel="Stages",
    no_args_is_help=True
)

default_config = dict(
    storage=Storages.local.value,
    token="",
    num_workers=1,
    chunksize=16,
    geocoder=Geocoders.local.value,
    dadata_api_key="",
    output_formats=["csv"],
    split_by_region=False,
    with_personal_names=False,
    with_crimea=True,
    with_new_territories=False,
)

app_dir = typer.get_app_dir(APP_NAME)
app_config_path = pathlib.Path(app_dir) / "config.json"
app_config_path.parent.mkdir(parents=True, exist_ok=True)
try:
    with open(app_config_path) as f:
        app_config = json.load(f)
except:
    app_config = {}
    print("Failed to load config, default options are loaded")
    app_config = default_config


def get_default_path(
    stage_name: str,
    source_dataset: Optional[str] = None,
    filename: Optional[str] = None,
) -> pathlib.Path:
    path = pathlib.Path("russian-sme-registry-data") / stage_name
    if source_dataset is not None:
        path = path / source_dataset
    if filename is not None:
        path = path / filename

    return path


def get_downloader(app_config: dict) -> Downloader:
    storage = app_config.get("storage")
    token = app_config.get("token")

    return Downloader(storage, token)


def get_extractor(app_config: dict) -> Extractor:
    num_workers = app_config.get("num_workers")
    chunksize = app_config.get("chunksize")
    storage = app_config.get("storage")
    token = app_config.get("token")

    return Extractor(storage, num_workers, chunksize, token)


@download_app.command("all", rich_help_panel="Source dataset(s)")
def download_all(
    download_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to the directory to store downloaded files. Sub-directories *sme*, *revexp*, and *empl* for respective datasets will be created automatically"
        )
    ] = get_default_path(StageNames.download.value)
):
    """
    Download all three source dataset(s)
    """
    d = get_downloader(app_config)
    for source_dataset in SourceDatasets:
        args = dict(
            download_dir=str(download_dir / source_dataset.value),
            source_dataset = source_dataset.value
        )
        d(**args)


@download_app.command("sme", rich_help_panel="Source dataset(s)")
def download_sme(
    download_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to the directory to store downloaded files"
        )
    ] = get_default_path(StageNames.download.value, SourceDatasets.sme.value)
):
    """
    Download **s**mall&**m**edium-sized **e**nterprises registry
    """
    d = get_downloader(app_config)
    d(SourceDatasets.sme.value, str(download_dir))


@download_app.command("revexp", rich_help_panel="Source dataset(s)")
def download_revexp(
    download_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to the directory to store downloaded files"
        )
    ] = get_default_path(StageNames.download.value, SourceDatasets.revexp.value)
):
    """
    Download data on **rev**enue and **exp**enditure of companies
    """
    d = get_downloader(app_config)
    d(SourceDatasets.revexp.value, str(download_dir))


@download_app.command("empl", rich_help_panel="Source dataset(s)")
def download_empl(
    download_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to the directory to store downloaded files"
        )
    ] = get_default_path(StageNames.download.value, SourceDatasets.empl.value)
):
    """
    Download data on number of **empl**oyees in companies
    """
    d = get_downloader(app_config)
    d(SourceDatasets.empl.value, str(download_dir))


@extract_app.command("all", rich_help_panel="Source dataset(s)")
def extract_all(
    in_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to downloaded source files. Usually the same as *download_dir* on download stage. Expected to contain *sme*, *revexp*, *empl* sub-folders"
        )
    ] = get_default_path(StageNames.download.value),
    out_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save extracted CSV files. Sub-folders *sme*, *revexp*, *empl* for respective datasets will be created automatically"
        )
    ] = get_default_path(StageNames.extract.value),
    clear: Annotated[
        bool, typer.Option(help="Clear *out_dir* (see above) before processing")
    ] = False,
    ac: Annotated[
        Optional[List[str]],
        typer.Option(
            help="**A**ctivity **c**ode(s) to filter SME source dataset by. Can be either activity group code, e.g. *--ac A*, or exact digit code, e.g. *--ac 01.10*. Multiple codes or groups can be specified by multiple *ac* options, e.g. *--ac 01.10 --ac 69.20*. Top-level codes include child codes, i.e. *--ac 01.10* selects 01.10.01, 01.10.02, 01.10.10 (if any children are present). If not specified, filtering is disabled",
            show_default="no filtering by activity code(s)"
        )
    ] = None,
):
    """
    Extract data from all three downloaded source datasets
    """
    e = get_extractor(app_config)
    for source_dataset in SourceDatasets:
        args = dict(
            in_dir=str(in_dir / source_dataset.value),
            out_dir=str(out_dir / source_dataset.value),
            source_dataset=source_dataset.value,
            clear=clear,
            activity_codes=ac,
        )
        e(**args)


@extract_app.command("sme", rich_help_panel="Source dataset(s)")
def extract_sme(
    in_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to downloaded source files. Usually the same as *download_dir* on download stage"
        )
    ] = get_default_path(StageNames.download.value, SourceDatasets.sme.value),
    out_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save extracted CSV files"
        )
    ] = get_default_path(StageNames.extract.value, SourceDatasets.sme.value),
    clear: Annotated[
        bool, typer.Option(help="Clear *out_dir* (see above) before processing")
    ] = False,
    ac: Annotated[
        Optional[List[str]],
        typer.Option(
            help="**A**ctivity **c**ode(s) to filter SME source dataset by. Can be either activity group code, e.g. *--ac A*, or exact digit code, e.g. *--ac 01.10*. Multiple codes or groups can be specified by multiple *ac* options, e.g. *--ac 01.10 --ac 69.20*. Top-level codes include child codes, i.e. *--ac 01.10* selects 01.10.01, 01.10.02, 01.10.10 (if any children are present). If not specified, filtering is disabled",
            show_default="no filtering by activity code(s)"
        )
    ] = None,
):
    """
    Extract data from downloaded *zip* archives of SME registry to *csv* files,
    optionally filtering by activity code (stage 2)
    """
    e = get_extractor(app_config)
    e(str(in_dir), str(out_dir), SourceDatasets.sme.value, clear, ac)


@extract_app.command("revexp", rich_help_panel="Source dataset(s)")
def extract_revexp(
    in_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to downloaded source files. Usually the same as *download_dir* on download stage"
        )
    ] = get_default_path(StageNames.download.value, SourceDatasets.revexp.value),
    out_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save extracted CSV files"
        )
    ] = get_default_path(StageNames.extract.value, SourceDatasets.revexp.value),
    clear: Annotated[
        bool, typer.Option(help="Clear *out_dir* (see above) before processing")
    ] = False
):
    """
    Extract data from downloaded *zip* archives of revexp data to *csv* files
    """
    e = get_extractor(app_config)
    e(str(in_dir), str(out_dir), SourceDatasets.revexp.value, clear)


@extract_app.command("empl", rich_help_panel="Source dataset(s)")
def extract_empl(
    in_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to downloaded source files. Usually the same as *download_dir* on download stage"
        )
    ] = get_default_path(StageNames.download.value, SourceDatasets.empl.value),
    out_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save extracted CSV files"
        )
    ] = get_default_path(StageNames.extract.value, SourceDatasets.empl.value),
    clear: Annotated[
        bool, typer.Option(help="Clear *out_dir* (see above) before processing")
    ] = False
):
    """
    Extract data from downloaded *zip* archives of empl data to *csv* files
    """
    e = get_extractor(app_config)
    e(str(in_dir), str(out_dir), SourceDatasets.empl.value, clear)


@aggregate_app.command("all", rich_help_panel="Source dataset(s)")
def aggregate_all(
    in_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to extracted CSV files. Usually the same as *out_dir* on extract stage. Expected to contain *sme*, *revexp*, *empl* sub-folders"
        )
    ] = get_default_path(StageNames.extract.value),
    out_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save aggregated CSV files. Sub-folders *sme*, *revexp*, *empl* for respective datasets will be created automatically"
        )
    ] = get_default_path(StageNames.aggregate.value),
):
    """
    Aggregate all three source datasets
    """
    a = Aggregator()
    for source_dataset in SourceDatasets:
        args = dict(
            in_dir=str(in_dir / source_dataset.value),
            out_file=str(out_dir / source_dataset.value / "agg.csv"),
            source_dataset=source_dataset.value,
        )
        if source_dataset.value == SourceDatasets.sme.value:
            args["with_crimea"] = app_config.get("with_crimea", True)
            args["with_new_territories"] = app_config.get("with_new_territories", False)

        if source_dataset.value in ("revexp", "empl"):
            args["sme_data_file"] = str(out_dir / SourceDatasets.sme.value / "agg.csv")
        a(**args)


@aggregate_app.command("sme", rich_help_panel="Source dataset(s)")
def aggregate_sme(
    in_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to extracted CSV files. Usually the same as *out_dir* on extract stage"
        )
    ] = get_default_path(StageNames.extract.value, SourceDatasets.sme.value),
    out_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save aggregated CSV files"
        )
    ] = get_default_path(StageNames.aggregate.value, SourceDatasets.sme.value, "agg.csv"),
):
    """
    Aggregate SME dataset
    """
    a = Aggregator()
    a(
        str(in_dir),
        str(out_file),
        SourceDatasets.sme.value,
        with_crimea=app_config.get("with_crimea", True),
        with_new_territories=app_config.get("with_new_territories", False)
    )


@aggregate_app.command("revexp", rich_help_panel="Source dataset(s)")
def aggregate_revexp(
    in_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to extracted CSV files. Usually the same as *out_dir* on extract stage"
        )
    ] = get_default_path(StageNames.extract.value, SourceDatasets.revexp.value),
    out_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save aggregated CSV files"
        )
    ] = get_default_path(StageNames.aggregate.value, SourceDatasets.revexp.value, "agg.csv"),
    sme_data_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to **already processed SME file** that is used to filter aggregated values in revexp or empl file",
            exists=True,
            file_okay=True,
            readable=True
        )
    ] = None
):
    """
    Aggregate revexp dataset
    """
    a = Aggregator()
    if sme_data_file is not None:
        sme_data_file = str(sme_data_file)

    a(str(in_dir), str(out_file), SourceDatasets.revexp.value, sme_data_file)


@aggregate_app.command("empl", rich_help_panel="Source dataset(s)")
def aggregate_empl(
    in_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to extracted CSV files. Usually the same as *out_dir* on extract stage"
        )
    ] = get_default_path(StageNames.extract.value, SourceDatasets.empl.value),
    out_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save aggregated CSV files"
        )
    ] = get_default_path(StageNames.aggregate.value, SourceDatasets.empl.value, "agg.csv"),
    sme_data_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to **already processed SME file** that is used to filter aggregated values in revexp or empl file",
            exists=True,
            file_okay=True,
            readable=True
        )
    ] = None
):
    """
    Aggregate empl dataset
    """
    a = Aggregator()
    if sme_data_file is not None:
        sme_data_file = str(sme_data_file)

    a(str(in_dir), str(out_file), SourceDatasets.empl.value, sme_data_file)


@app.command(rich_help_panel="Stages")
def geocode(
    in_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to a CSV file. Usually the same as *out_file* on aggregate sme stage. CSV file must include at least *region_name*, *district_name*, *city_name*, *settlement_name* fields",
            exists=True,
            file_okay=True,
            readable=True
        )
    ] = get_default_path(StageNames.aggregate.value, SourceDatasets.sme.value, "agg.csv"),
    out_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save geocoded CSV file"
        )
    ] = get_default_path(StageNames.geocode.value, SourceDatasets.sme.value, "geocoded.csv"),
):
    """
    Geocode SME aggregated data (stage 4) OR geocode *any other dataset with a similar structure of address fields*
    """
    if app_config.get("geocoder") == Geocoders.dadata.value:
        g = DaDataGeocoder(app_config.get("dadata_api_key"))
    else:
        g = LocalGeocoder()
    g(str(in_file), str(out_file))


@app.command(rich_help_panel="Stages")
def panelize(
    sme_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to geocoded CSV file. Usually the same as *out_file* on geocode stage",
            exists=True,
            file_okay=True,
            readable=True
        )
    ] = get_default_path(StageNames.geocode.value, SourceDatasets.sme.value, "geocoded.csv"),
    out_dir: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to save panel CSV file"
        )
    ] = get_default_path(StageNames.panelize.value),
    revexp_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to aggregated CSV revexp file. Usually the same as *out_file* on aggregate revexp stage",
            exists=True,
            file_okay=True,
            readable=True
        )
    ] = get_default_path(StageNames.aggregate.value, SourceDatasets.revexp.value, "agg.csv"),
    empl_file: Annotated[
        Optional[pathlib.Path],
        typer.Option(
            help="Path to aggregated CSV empl file. Usually the same as *out_file* on aggregate empl stage",
            exists=True,
            file_okay=True,
            readable=True
        )
    ] = get_default_path(StageNames.aggregate.value, SourceDatasets.empl.value, "agg.csv"),
):
    """
    Make panel dataset based on geocoded SME data and aggregated revexp and empl tables (stage 5)
    """
    remove_personal_names = app_config.get("remove_personal_names", True)
    save_to_csv = "csv" in app_config.get("output_formats", [])
    save_to_parquet = "parquet" in app_config.get("output_formats", [])
    save_to_excel = "excel" in app_config.get("output_formats", [])
    split_by_region = app_config.get("split_by_region", False)

    p = Panelizer()
    p(
        str(sme_file),
        str(out_dir),
        str(revexp_file),
        str(empl_file),
        remove_personal_names,
        save_to_csv,
        save_to_parquet,
        save_to_excel,
        split_by_region,
    )


@app.command(rich_help_panel="Configuration", no_args_is_help=True)
def config(
    show: Annotated[
        bool,
        typer.Option(
            "--show",
            help="Only show current config without updating",
            show_default="false",
            rich_help_panel="Control")
    ] = False,
    chunksize: Annotated[
        Optional[int],
        typer.Option(
            help="Chunk size for extractor",
            rich_help_panel="Available options",
            show_default="16"
        )
    ] = None,
    num_workers: Annotated[
        Optional[int],
        typer.Option(
            help="Number of workers = processes for extractor. "
                 "Bigger is faster, but cannot be higher than number of CPU cores",
            rich_help_panel="Available options",
            show_default="1"
        )
    ] = None,
    storage: Annotated[
        Optional[Storages],
        typer.Option(
            help="Place to store downloaded source datasets "
                 " and/or to later get them for extract stage. "
                 " Note that ydisk_public option is used by extractor only"
                 " and causes error on download stage",
            rich_help_panel="Available options",
            show_default="local"
        )
    ] = None,
    token: Annotated[
        Optional[str],
        typer.Option(
            help="Token for Yandex Disk; used if *storage* is *ydisk*",
            rich_help_panel="Available options",
            show_default="<empty>"
        )
    ] = None,
    geocoder: Annotated[
        Optional[Geocoders],
        typer.Option(
            help="Geocoder to use",
            rich_help_panel="Available options",
            show_default="local"
        )
    ] = None,
    dadata_api_key: Annotated[
        Optional[str],
        typer.Option(
            help="API key for DaData geocoder",
            rich_help_panel="Available options",
            show_default="<empty>"
        )
    ] = None,
    output_formats: Annotated[
        Optional[List[str]],
        typer.Option(
            help="Output formats to save panel dataset to. Can be *csv*, *parquet*, *excel*. Multiple formats can be specified by multiple *output_formats* options, e.g. *--output_formats csv parquet*",
            rich_help_panel="Available options",
            show_default="csv",
        )
    ] = None,
    split_by_region: Annotated[
        Optional[bool],
        typer.Option(
            "--split-by-region/--combine-regions",
            help="Split panel dataset by region",
            rich_help_panel="Available options",
            show_default="false"
        )
    ] = None,
    with_personal_names: Annotated[
        Optional[bool],
        typer.Option(
            "--with-personal-names/--remove-personal-names",
            help="Remove personal names of sole traders from panel dataset for privacy reasons",
            rich_help_panel="Available options",
            show_default="false"
        )
    ] = None,
    with_crimea: Annotated[
        Optional[bool],
        typer.Option(
            "--with-crimea/--exclude-crimea",
            help="Include Crimea and Sevastopol regions",
            rich_help_panel="Available options",
            show_default="true"
        )
    ] = None,
    with_new_territories: Annotated[
        Optional[bool],
        typer.Option(
            "--with-new-territories/--exclude-new-territories",
            help="Include Donetsk, Luhansk, Zaporozhye, and Kherson regions",
            rich_help_panel="Available options",
            show_default="false"
        )
    ] = None,
):
    """
    Show or set global options for all commands
    """
    if show:
        print_config(app_config)
        return

    if num_workers and num_workers > os.cpu_count():
        num_workers = os.cpu_count()
        print(f"Number of workers is set to {num_workers} (max number of CPU cores)")

    loc = locals()
    new_config = {}
    for param_name in (
        "token",
        "num_workers",
        "chunksize",
        "storage",
        "geocoder",
        "dadata_api_key",
        "output_formats",
        "split_by_region",
        "with_personal_names",
        "with_crimea",
        "with_new_territories",
    ):
        current_value = app_config.get(param_name)
        new_value = loc.get(param_name)
        if new_value is not None and new_value != []:
            value = new_value
            if param_name in ("storage", "geocoder"):  # enums
                value = value.value
        elif current_value is None or current_value == []:
            value = default_config[param_name]
        else:
            value = app_config[param_name]

        new_config[param_name] = value

    with open(app_config_path, "w") as f:
        json.dump(new_config, f)
        app_config.update(new_config)

    print("Configuration updated")
    print_config(new_config)


@app.command(rich_help_panel="Magic command")
def process(
    download: Annotated[
        bool,
        typer.Option(
            help="Download source datasets before processing. If False, the application expects that source datasets have already been downloaded to *rmsp-data/download/sme*, *rmsp-data/download/revexp*, and rmsp-data/download/empl*"
        )
    ] = False,
    ac: Annotated[
        Optional[List[str]],
        typer.Option(
            help="**A**ctivity **c**ode(s) to filter SME source dataset by. Can be either activity group code, e.g. *--ac A*, or exact digit code, e.g. *--ac 01.10*. Multiple codes or groups can be specified by multiple *ac* options, e.g. *--ac 01.10 --ac 69.20*. Top-level codes include child codes, i.e. *--ac 01.10* selects 01.10.01, 01.10.02, 01.10.10 (if any children are present). If not specified, filtering is disabled",
            show_default="no filtering by activity code(s)"
        )
    ] = None,
):
    """
    Process the source data with this single command
    """
    if download:
        download_all()

    extract_all(ac=ac)
    aggregate_all()
    geocode()
    panelize()
