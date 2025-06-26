import pathlib
import shutil

import pandas as pd
import requests
from typer.testing import CliRunner

from russian_sme_registry.main import app, app_config
from russian_sme_registry.stages.download import Downloader
from russian_sme_registry.utils.enums import SourceDatasets
from .common import mock_get, mock_post, mock_put, mock_ydisk_api

runner = CliRunner()


def test_help():
    """Test help command works"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_download_all(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["download", "all"])

    assert result.exit_code == 0

    download_subdirs = list((tmp_path / "russian-sme-registry-data" / "download").iterdir())
    assert len(download_subdirs) == 3
    assert sorted(d.name for d in download_subdirs) == ["empl", "revexp", "sme"]

    sme_files = list((tmp_path / "russian-sme-registry-data" / "download" / "sme").iterdir())
    assert len(sme_files) > 1
    assert sme_files[0].name.startswith("data-")
    assert sme_files[0].name.endswith(".zip")

    revexp_files = list((tmp_path / "russian-sme-registry-data" / "download" / "revexp").iterdir())
    assert len(revexp_files) == 1
    assert revexp_files[0].name.startswith("data-")
    assert revexp_files[0].name.endswith(".zip")

    empl_files = list((tmp_path / "russian-sme-registry-data" / "download" / "empl").iterdir())
    assert len(empl_files) == 1
    assert empl_files[0].name.startswith("data-")
    assert empl_files[0].name.endswith(".zip")


def test_download_all_custom_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["download", "all", "--download-dir", "custom-dir"])

    assert result.exit_code == 0

    download_subdirs = list((tmp_path / "custom-dir").iterdir())
    assert len(download_subdirs) == 3
    assert sorted(d.name for d in download_subdirs) == ["empl", "revexp", "sme"]


def test_extract_all(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    tests_data_dir = pathlib.Path(__file__).parent / "data"
    for source_dataset in SourceDatasets:
        for f in tests_data_dir.glob(f"{source_dataset.value}/*.zip"):
            target_dir = tmp_path / "russian-sme-registry-data" / "download" / source_dataset.value
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, target_dir / f.name)

    result = runner.invoke(app, ["extract", "all"])

    assert result.exit_code == 0

    assert len(list((tmp_path / "russian-sme-registry-data" / "extract").iterdir())) == 3
    assert len(list((tmp_path / "russian-sme-registry-data" / "extract" / "sme").glob("*.csv"))) == 2
    assert len(list((tmp_path / "russian-sme-registry-data" / "extract" / "revexp").glob("*.csv"))) == 2
    assert len(list((tmp_path / "russian-sme-registry-data" / "extract" / "empl").glob("*.csv"))) == 2


def test_extract_all_custom_dirs(monkeypatch, tmp_path):
    args = ["extract", "all"]
    args.extend(["--in-dir", str(pathlib.Path(__file__).parent / "data")])
    args.extend(["--out-dir", str(tmp_path / "results")])

    result = runner.invoke(app, args)

    assert result.exit_code == 0

    assert len(list((tmp_path / "results").iterdir())) == 3
    assert len(list((tmp_path / "results" / "sme").glob("*.csv"))) == 2
    assert len(list((tmp_path / "results" / "revexp").glob("*.csv"))) == 2
    assert len(list((tmp_path / "results" / "empl").glob("*.csv"))) == 2


def test_extract_all_clear_out_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")

    for option in ("sme", "revexp", "empl"):
        dir = tmp_path / "results" / option
        dir.mkdir(parents=True)
        for j in range(100):
            (dir / f"file-{j}.csv").touch()

    args = ["extract", "all"]
    args.extend(["--in-dir", str(pathlib.Path(__file__).parent / "data")])
    args.extend(["--out-dir", str(tmp_path / "results")])
    args.extend(["--clear"])
    result = runner.invoke(app, args)

    assert result.exit_code == 0

    assert len(list((tmp_path / "results").iterdir())) == 3

    # All generated empty CSV files should be removed by --clear
    assert len(list((tmp_path / "results" / "sme").glob("*.csv"))) == 2
    assert len(list((tmp_path / "results" / "revexp").glob("*.csv"))) == 2
    assert len(list((tmp_path / "results" / "empl").glob("*.csv"))) == 2


def test_extract_all_filter_by_activity_code(monkeypatch, tmp_path):
    out_dir = tmp_path / "results"

    args = ["extract", "all"]
    args.extend(["--in-dir", str(pathlib.Path(__file__).parent / "data")])
    args.extend(["--out-dir", str(out_dir)])
    args.extend(["--ac", "47", "--ac", "49"])

    result = runner.invoke(app, args)

    assert result.exit_code == 0

    assert len(list(out_dir.iterdir())) == 3

    extracted = pd.read_csv(out_dir / "sme" / "data-test-sme-1.csv", dtype=str)
    assert all(
        c.startswith("47") or c.startswith("49")
        for c in extracted["activity_code_main"].unique()
    )
    assert len(extracted.loc[extracted["ind_tin"] == "523102417490"]) == 1
    assert len(extracted.loc[extracted["ind_tin"] == "523400533580"]) == 1


def test_extract_revexp_empl_does_not_accept_activity_code():
    result = runner.invoke(app, ["extract", "revexp", "--ac", "47"])
    assert result.exit_code != 0
    assert "No such option" in result.stdout

    result = runner.invoke(app, ["extract", "empl", "--ac", "47"])
    assert result.exit_code != 0
    assert "No such option" in result.stdout


def test_aggregate_all(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    in_dir = tmp_path / "russian-sme-registry-data" / "extract"
    data_dir = pathlib.Path(__file__).parent / "data"
    for source_dataset in SourceDatasets:
        target_dir = in_dir / source_dataset.value
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in data_dir.glob(f"{source_dataset.value}/data*extracted*.csv"):
            shutil.copy(f, target_dir / f.name)

    result = runner.invoke(app, ["aggregate", "all"])

    assert result.exit_code == 0

    out_dir = tmp_path / "russian-sme-registry-data" / "aggregate"
    assert len(list(out_dir.iterdir())) == 3
    for source_dataset in SourceDatasets:
        assert (out_dir / source_dataset.value / "agg.csv").exists()


def test_aggregate_all_custom_dirs(tmp_path):
    in_dir = pathlib.Path(__file__).parent / "data"
    out_dir = tmp_path / "results"

    args = ["aggregate", "all"]
    args.extend(["--in-dir", str(in_dir)])
    args.extend(["--out-dir", str(out_dir)])

    result = runner.invoke(app, args)

    assert result.exit_code == 0

    assert len(list(out_dir.iterdir())) == 3
    for source_dataset in SourceDatasets:
        assert (out_dir / source_dataset.value / "agg.csv").exists()


def test_aggregate_revexp_filtering_option(tmp_path):
    in_dir = pathlib.Path(__file__).parent / "data"
    out_dir = tmp_path / "results"

    args = ["aggregate", "revexp"]
    args.extend(["--in-dir", str(in_dir / "revexp")])
    args.extend(["--out-file", str(out_dir / "revexp" / "agg.csv")])

    result = runner.invoke(app, args)
    assert result.exit_code == 0

    df = pd.read_csv(out_dir / "revexp" / "agg.csv")
    assert len(df) > 0

    args.extend(["--sme-data-file", str(in_dir / "sme" / "test-aggregated.csv")])

    result = runner.invoke(app, args)

    assert result.exit_code == 0

    df = pd.read_csv(out_dir / "revexp" / "agg.csv")
    assert len(df) == 0


def test_aggregate_empl_filtering_option(tmp_path):
    in_dir = pathlib.Path(__file__).parent / "data"
    out_dir = tmp_path / "results"

    args = ["aggregate", "empl"]
    args.extend(["--in-dir", str(in_dir / "empl")])
    args.extend(["--out-file", str(out_dir / "empl" / "agg.csv")])

    result = runner.invoke(app, args)
    assert result.exit_code == 0

    df = pd.read_csv(out_dir / "empl" / "agg.csv")
    assert len(df) > 2

    args.extend(["--sme-data-file", str(in_dir / "sme" / "test-aggregated.csv")])

    result = runner.invoke(app, args)

    assert result.exit_code == 0

    df = pd.read_csv(out_dir / "empl" / "agg.csv")
    assert len(df) == 2


def test_geocode(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    src = pathlib.Path(__file__).parent / "data" / "sme" / "test-aggregated.csv"
    dst = tmp_path / "russian-sme-registry-data" / "aggregate" / "sme" / "agg.csv"
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)

    result = runner.invoke(app, ["geocode"])
    assert result.exit_code == 0

    out_file = tmp_path / "russian-sme-registry-data" / "geocode" / "sme" / "geocoded.csv"
    assert out_file.exists()


def test_geocode_custom_paths(tmp_path):
    in_file = pathlib.Path(__file__).parent / "data" / "sme" / "test-aggregated.csv"
    out_file = tmp_path /  "geocoded.csv"

    args = ["geocode", "--in-file", str(in_file), "--out-file", str(out_file)]

    result = runner.invoke(app, args)
    assert result.exit_code == 0

    assert out_file.exists()


def test_panelize(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    sme_in_file_src = pathlib.Path(__file__).parent / "data" / "sme" / "test-geocoded.csv"
    sme_in_file_dst = tmp_path / "russian-sme-registry-data" / "geocode" / "sme" / "geocoded.csv"
    sme_in_file_dst.parent.mkdir(parents=True)
    shutil.copy(sme_in_file_src, sme_in_file_dst)

    revexp_in_file_src = pathlib.Path(__file__).parent / "data" / "revexp" / "test-aggregated.csv"
    revexp_in_file_dst = tmp_path / "russian-sme-registry-data" / "aggregate" / "revexp" / "agg.csv"
    revexp_in_file_dst.parent.mkdir(parents=True)
    shutil.copy(revexp_in_file_src, revexp_in_file_dst)

    empl_in_file_src = pathlib.Path(__file__).parent / "data" / "empl" / "test-aggregated.csv"
    empl_in_file_dst = tmp_path / "russian-sme-registry-data" / "aggregate" / "empl" / "agg.csv"
    empl_in_file_dst.parent.mkdir(parents=True)
    shutil.copy(empl_in_file_src, empl_in_file_dst)

    result = runner.invoke(app, ["panelize"])
    assert result.exit_code == 0

    assert (tmp_path / "russian-sme-registry-data" / "panelize" / "csv" / "all-regions.csv").exists()


def test_panelize_custom_paths(tmp_path):
    sme_file = pathlib.Path(__file__).parent / "data" / "sme" / "test-geocoded.csv"
    revexp_file = pathlib.Path(__file__).parent / "data" / "revexp" / "test-aggregated.csv"
    empl_file = pathlib.Path(__file__).parent / "data" / "empl" / "test-aggregated.csv"

    sme_out_file = tmp_path / "custom" / "csv" / "all-regions.csv"

    args = [
        "panelize",
        "--sme-file",
        str(sme_file),
        "--revexp-file",
        str(revexp_file),
        "--empl-file",
        str(empl_file),
        "--out-dir",
        str(tmp_path / "custom"),
    ]

    result = runner.invoke(app, args)
    assert result.exit_code == 0

    assert sme_out_file.exists()


def test_process_dry_run(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["process"])

    assert result.exit_code == 0


def test_process_no_download(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    for source_dataset in SourceDatasets:
        src = pathlib.Path(__file__).parent / "data" / source_dataset.value
        dst = tmp_path / "russian-sme-registry-data" / "download" / source_dataset.value
        dst.mkdir(parents=True)
        for f in src.glob("*.zip"):
            shutil.copy(f, dst / f.name)

    result = runner.invoke(app, ["process"])
    assert result.exit_code == 0

    output_dir = tmp_path / "russian-sme-registry-data"

    assert len(list((output_dir / "extract").iterdir())) == 3
    assert len(list((output_dir / "aggregate").iterdir())) == 3
    assert len(list((output_dir / "geocode").iterdir())) == 1
    assert (output_dir / "panelize" / "csv" / "all-regions.csv").exists()

def test_process_with_download(monkeypatch, tmp_path):
    def mock_get_data_urls(self, source_dataset: str):
        data_dir = pathlib.Path(__file__).parent / "data" / source_dataset
        return [
            f"data/{source_dataset}/{f.name}" for f in data_dir.glob("*.zip")
        ]

    monkeypatch.setattr(Downloader, "_get_data_urls", mock_get_data_urls)
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["process", "--download"])
    assert result.exit_code == 0

    output_dir = tmp_path / "russian-sme-registry-data"

    assert len(list((output_dir / "download").iterdir())) == 3
    assert len(list((output_dir / "extract").iterdir())) == 3
    assert len(list((output_dir / "aggregate").iterdir())) == 3
    assert len(list((output_dir / "geocode").iterdir())) == 1
    assert (output_dir / "panelize" / "csv" / "all-regions.csv").exists()


def test_config():
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0

    args = ["config", "--token", "test-token", "--storage", "ydisk", "--chunksize", "1024"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "Configuration updated" in result.stdout

    result = runner.invoke(app, ["config", "--show"])
    assert result.exit_code == 0
    assert "test-token" in result.stdout
    assert "ydisk" in result.stdout
    assert "1024" in result.stdout

    args = ["config", "--storage", "local"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert "Configuration updated" in result.stdout

    result = runner.invoke(app, ["config", "--show"])
    assert result.exit_code == 0
    assert "local" in result.stdout
    assert "1024" in result.stdout # chunksize is not changed


def test_download_ydisk(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(requests, "put", mock_put)
    monkeypatch.setattr(Downloader, "YDISK_DOWNLOAD_TIMEOUT", 0)
    monkeypatch.chdir(tmp_path)

    mock_ydisk_api.clear()

    with monkeypatch.context() as m:
        m.setitem(app_config, "storage", "ydisk")
        m.setitem(app_config, "ydisk_token", "token")

        result = runner.invoke(app, ["download", "all"])

    assert result.exit_code == 0
    assert result.stdout.count("Upload task created successfully") == 105


def test_extract_ydisk(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(requests, "put", mock_put)
    monkeypatch.setattr(Downloader, "YDISK_DOWNLOAD_TIMEOUT", 0)
    monkeypatch.chdir(tmp_path)

    mock_ydisk_api.clear()
    for source_dataset in SourceDatasets:
        mock_ydisk_api.put(
            "disk/resources",
            dict(Authorization=f"OAuth token"),
            dict(path=source_dataset.value),
        )

        data_dir = pathlib.Path(__file__).parent / "data" / source_dataset.value
        for fn in data_dir.glob("*.zip"):
            mock_ydisk_api.post(
                "disk/resources/upload",
                dict(Authorization=f"OAuth token"),
                dict(
                    path=f"{source_dataset.value}/{fn.name}",
                    url=fn.name
                ),
            )

    with monkeypatch.context() as m:
        m.setitem(app_config, "storage", "ydisk")
        m.setitem(app_config, "ydisk_token", "token")

        result = runner.invoke(app, ["extract", "all", "--in-dir", "."])

    assert result.exit_code == 0
    assert result.stdout.count("Completed in") == 6

    assert len(list((tmp_path / "russian-sme-registry-data" / "extract").iterdir())) == 3
    assert len(list((tmp_path / "russian-sme-registry-data" / "extract" / "sme").glob("*.csv"))) == 2
    assert len(list((tmp_path / "russian-sme-registry-data" / "extract" / "revexp").glob("*.csv"))) == 2
    assert len(list((tmp_path / "russian-sme-registry-data" / "extract" / "empl").glob("*.csv"))) == 2


def test_process_ydisk_no_download(monkeypatch, tmp_path):
    monkeypatch.setattr(requests, "get", mock_get)
    monkeypatch.setattr(requests, "post", mock_post)
    monkeypatch.setattr(requests, "put", mock_put)
    monkeypatch.setattr(Downloader, "YDISK_DOWNLOAD_TIMEOUT", 0)
    monkeypatch.chdir(tmp_path)

    mock_ydisk_api.clear()
    for top_level_path in ("russian-sme-registry-data", "russian-sme-registry-data/download"):
        mock_ydisk_api.put(
            "disk/resources",
            dict(Authorization=f"OAuth token"),
            dict(path=top_level_path),
        )

    for source_dataset in SourceDatasets:
        mock_ydisk_api.put(
            "disk/resources",
            dict(Authorization=f"OAuth token"),
            dict(path=f"russian-sme-registry-data/download/{source_dataset.value}"),
        )

        data_dir = pathlib.Path(__file__).parent / "data" / source_dataset.value
        upload_dir = f"russian-sme-registry-data/download/{source_dataset.value}"
        for fn in data_dir.glob("*.zip"):
            mock_ydisk_api.post(
                "disk/resources/upload",
                dict(Authorization=f"OAuth token"),
                dict(
                    path=f"{upload_dir}/{fn.name}",
                    url=fn.name
                ),
            )

    with monkeypatch.context() as m:
        m.setitem(app_config, "storage", "ydisk")
        m.setitem(app_config, "ydisk_token", "token")

        result = runner.invoke(app, ["process"])

    assert result.exit_code == 0
    assert result.stdout.count("Completed in") == 6

    assert len(list((tmp_path / "russian-sme-registry-data" / "extract").iterdir())) == 3
    assert len(list((tmp_path / "russian-sme-registry-data" / "aggregate").iterdir())) == 3
    assert len(list((tmp_path / "russian-sme-registry-data" / "geocode").iterdir())) == 1
    assert (tmp_path / "russian-sme-registry-data" / "panelize" / "csv" / "all-regions.csv").exists()
