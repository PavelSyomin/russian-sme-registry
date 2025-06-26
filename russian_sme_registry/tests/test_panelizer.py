import pathlib

import numpy as np
import pandas as pd
import pytest

from ..stages.panelize import Panelizer


def test_panelize(tmp_path):
    sme_file = pathlib.Path(__file__).parent / "data/sme/test-geocoded.csv"
    revexp_file = pathlib.Path(__file__).parent / "data/revexp/test-aggregated.csv"
    empl_file = pathlib.Path(__file__).parent / "data/empl/test-aggregated.csv"
    out_file = tmp_path / "csv/all-regions.csv"

    panelizer = Panelizer()

    panelizer(str(sme_file), str(tmp_path), str(revexp_file), str(empl_file))

    df = pd.read_csv(out_file, dtype=str, sep=";")

    assert len(df) == 58
    assert "year" in df.columns
    assert "tax_number" in df.columns

    assert df["revenue"].notna().sum() == 0
    assert df["expenditure"].notna().sum() == 0
    assert df["employees_count"].notna().sum() == 2

    with_empl_count = df.loc[df["employees_count"].notna()]
    assert list(with_empl_count["tax_number"].unique()) == ["8901037297"]
    assert list(with_empl_count["year"].unique()) == ["2019"]


def test_panelize_save_to_parquet(tmp_path):
    sme_file = pathlib.Path(__file__).parent / "data/sme/test-geocoded.csv"
    revexp_file = pathlib.Path(__file__).parent / "data/revexp/test-aggregated.csv"
    empl_file = pathlib.Path(__file__).parent / "data/empl/test-aggregated.csv"
    out_file = tmp_path / "parquet/all-regions.parquet"

    panelizer = Panelizer()

    panelizer(str(sme_file), str(tmp_path), str(revexp_file), str(empl_file), save_to_csv=False, save_to_parquet=True)

    df = pd.read_parquet(out_file)

    assert len(df) == 58
    assert "year" in df.columns
    assert "tax_number" in df.columns

    assert df["revenue"].notna().sum() == 0
    assert df["expenditure"].notna().sum() == 0
    assert df["employees_count"].notna().sum() == 2

    with_empl_count = df.loc[df["employees_count"].notna()]
    assert list(with_empl_count["tax_number"].unique()) == ["8901037297"]
    assert list(with_empl_count["year"].unique()) == [2019]


def test_panelize_save_to_excel(tmp_path):
    sme_file = pathlib.Path(__file__).parent / "data/sme/test-geocoded.csv"
    revexp_file = pathlib.Path(__file__).parent / "data/revexp/test-aggregated.csv"
    empl_file = pathlib.Path(__file__).parent / "data/empl/test-aggregated.csv"
    out_file = tmp_path / "excel/all-regions.xlsx"

    panelizer = Panelizer()

    panelizer(str(sme_file), str(tmp_path), str(revexp_file), str(empl_file), save_to_csv=False, save_to_excel=True)

    df = pd.read_excel(out_file)

    assert len(df) == 58
    assert "year" in df.columns
    assert "tax_number" in df.columns

    assert df["revenue"].notna().sum() == 0
    assert df["expenditure"].notna().sum() == 0
    assert df["employees_count"].notna().sum() == 2

    with_empl_count = df.loc[df["employees_count"].notna()]
    assert list(with_empl_count["tax_number"].unique()) == [8901037297]
    assert list(with_empl_count["year"].unique()) == [2019]


def test_panelize_no_revexp_empl(tmp_path):
    sme_file = pathlib.Path(__file__).parent / "data/sme/test-geocoded.csv"
    out_file = tmp_path / "csv/all-regions.csv"

    panelizer = Panelizer()

    panelizer(str(sme_file), str(tmp_path))

    df = pd.read_csv(out_file, dtype=str, sep=";")

    assert len(df) == 58
    assert "year" in df.columns
    assert "tax_number" in df.columns

    for col in ("revenue", "expenditure", "employees_count"):
        assert col not in df.columns


def test_panelize_remove_personal_names(tmp_path):
    sme_file = pathlib.Path(__file__).parent / "data/sme/test-geocoded.csv"
    out_file = tmp_path / "csv/all-regions.csv"

    panelizer = Panelizer()

    panelizer(str(sme_file), str(tmp_path), remove_personal_names=True)

    df = pd.read_csv(out_file, dtype=str, sep=";")

    assert len(df) == 58
    assert "year" in df.columns
    assert "tax_number" in df.columns

    for col in ("first_name", "last_name", "patronymic"):
        assert col not in df.columns