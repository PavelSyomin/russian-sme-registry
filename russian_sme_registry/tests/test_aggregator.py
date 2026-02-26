import pathlib

import numpy as np
import pandas as pd
import pytest

from ..stages.aggregate import Aggregator
from ..utils.enums import SourceDatasets


def test_wrong_dataset_type(tmp_path):
    in_dir = tmp_path / "input_data"
    out_file = tmp_path / "aggregated.csv"
    in_dir.mkdir()

    aggregator = Aggregator()

    with pytest.raises(RuntimeError, match="source dataset"):
        aggregator(str(in_dir), str(out_file), "egrul")


def test_empty_input_directory(tmp_path):
    in_dir = tmp_path / "input_data"
    out_file = tmp_path / "aggregated.csv"
    in_dir.mkdir()

    aggregator = Aggregator()

    aggregator(str(in_dir), str(out_file), SourceDatasets.sme.value)
    assert not out_file.exists()

    aggregator(str(in_dir), str(out_file), SourceDatasets.revexp.value)
    assert not out_file.exists()

    aggregator(str(in_dir), str(out_file), SourceDatasets.empl.value)
    assert not out_file.exists()


def test_aggregate_sme(tmp_path):
    in_dir = pathlib.Path(__file__).parent / "data/sme"
    out_file = tmp_path / "aggregated.csv"

    aggregator = Aggregator()

    aggregator(str(in_dir), str(out_file), SourceDatasets.sme.value)

    df = pd.read_csv(out_file, dtype=str)

    assert len(df) == 37

    no_change = df.loc[df["tin"] == "591604272151"]
    assert len(no_change) == 1
    assert no_change["start_date"].iloc[0] == "2019-02-10"
    assert no_change["end_date"].iloc[0] == "2020-11-10"

    category_change = df.loc[df["tin"] == "8901037297"]
    assert len(category_change) == 2
    assert list(category_change["category"]) == ["1", "2"]
    assert list(category_change["start_date"]) == ["2019-02-10", "2019-05-10"]
    assert list(category_change["end_date"]) == ["2019-05-10", "2020-11-10"]

    last_name_change = df.loc[df["tin"] == "890102946212"]
    assert len(last_name_change) == 2
    assert list(last_name_change["last_name"]) == ["ГАЛЯМШИНА", "ГРОМОВА"]
    assert list(last_name_change["start_date"]) == ["2019-02-10", "2020-11-10"]
    assert list(last_name_change["end_date"]) == ["2020-11-10", "2020-11-10"]

    location_change = df.loc[df["tin"] == "5233003589"]
    assert len(location_change) == 2
    assert list(location_change["settlement_name"]) == ["ТОНКИНО", "ПИЖМА"]
    assert list(location_change["start_date"]) == ["2019-02-10", "2020-11-10"]
    assert list(location_change["end_date"]) == ["2020-11-10", "2020-11-10"]

    ac_change = df.loc[df["tin"] == "523201067705"]
    assert len(ac_change) == 2
    assert list(ac_change["activity_code_main"]) == ["49.42", "49.41"]
    assert list(ac_change["start_date"]) == ["2019-02-10", "2019-05-10"]
    assert list(ac_change["end_date"]) == ["2019-05-10", "2020-11-10"]

    reverted_change = df.loc[df["tin"] == "5235000135"]
    assert len(reverted_change) == 3
    assert list(reverted_change["org_short_name"]) == ["УРЕНСКОЕ РАЙПО", "УРЕНСКИЙ РАЙПОТРЕБ", "УРЕНСКОЕ РАЙПО"]
    assert list(reverted_change["start_date"]) == ["2019-02-10", "2019-05-10", "2020-11-10"]
    assert list(reverted_change["end_date"]) == ["2019-05-10", "2020-11-10", "2020-11-10"]

    location_case_change = df.loc[df["tin"] == "5233001662"]
    assert len(location_case_change) == 1
    assert list(location_case_change["start_date"]) == ["2019-02-10"]
    assert list(location_case_change["end_date"]) == ["2020-11-10"]

    multiple_simultaneous_changes = df.loc[df["tin"] == "523101378432"]
    assert len(multiple_simultaneous_changes) == 2
    assert list(multiple_simultaneous_changes["first_name"]) == ["АНДРЕЙ", "НИКОЛАЙ"]
    assert list(multiple_simultaneous_changes["activity_code_main"]) == ["16.10", "16.11"]
    assert list(multiple_simultaneous_changes["start_date"]) == ["2019-02-10", "2019-05-10"]
    assert list(multiple_simultaneous_changes["end_date"]) == ["2019-05-10", "2020-11-10"]

    multiple_consequtive_changes = df.loc[df["tin"] == "5234003790"]
    assert len(multiple_consequtive_changes) == 3
    assert list(multiple_consequtive_changes["org_name"]) == ["ПОТРЕБИТЕЛЬСКОЕ ОБЩЕСТВО \"ТОНШАЕВСКИЙ УНИВЕРСАМ\"", "ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ \"ТОНШАЕВСКИЙ УНИВЕРСАМ\"", "ПОТРЕБИТЕЛЬСКОЕ ОБЩЕСТВО \"ТОНШАЕВСКИЙ УНИВЕРСАМ\""]
    assert list(multiple_consequtive_changes["city_name"]) == [np.nan, np.nan, "НИЖНИЙ НОВГОРОД"]
    assert list(multiple_consequtive_changes["settlement_name"]) == ["ТОНШАЕВО", "ТОНШАЕВО", np.nan]
    assert list(multiple_consequtive_changes["start_date"]) == ["2019-02-10", "2019-05-10", "2020-11-10"]
    assert list(multiple_consequtive_changes["end_date"]) == ["2019-05-10", "2020-11-10", "2020-11-10"]


def test_aggregate_sme_gap_detection(tmp_path):
    """Test gaps-and-islands: TIN with records at 2019-02-10 and 2020-11-10 only
    (skipping 2019-05-10) should produce 2 rows because 2019-05-10 exists for other TINs.
    Each input file must have exactly one data_date (per-file validation).
    """
    in_dir = tmp_path / "sme_gap"
    in_dir.mkdir()
    out_file = tmp_path / "aggregated.csv"

    header = (
        '"kind","category","reestr_date","data_date","ind_tin","ind_number","first_name",'
        '"last_name","patronymic","org_name","org_short_name","org_tin","org_number",'
        '"region_code","region_name","region_type","district_name","district_type",'
        '"city_name","city_type","settlement_name","settlement_type","activity_code_main",'
        '"file_id","doc_cnt"\n'
    )
    # Split by date: each file has exactly one data_date (per-file validation)
    (in_dir / "data-sme-gap-201902.csv").write_text(
        header
        + '1,1,"10.12.2018","10.02.2019",,,,,,"ООО ""GAP TEST""","ООО GAP",9999999999,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_GAP_001",900\n'
        + '1,1,"10.12.2018","10.02.2019",,,,,,"ООО ""NO GAP""","ООО NG",7777777777,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_GAP_003",900\n',
        encoding="utf-8",
    )
    (in_dir / "data-sme-gap-201905.csv").write_text(
        header
        + '1,1,"10.12.2018","10.05.2019",,,,,,"ООО ""ANCHOR""","ООО A",8888888888,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_GAP_002",900\n'
        + '1,1,"10.12.2018","10.05.2019",,,,,,"ООО ""NO GAP""","ООО NG",7777777777,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_GAP_003",900\n',
        encoding="utf-8",
    )
    (in_dir / "data-sme-gap-202011.csv").write_text(
        header
        + '1,1,"10.12.2018","10.11.2020",,,,,,"ООО ""GAP TEST""","ООО GAP",9999999999,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_GAP_001",900\n'
        + '1,1,"10.12.2018","10.11.2020",,,,,,"ООО ""NO GAP""","ООО NG",7777777777,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_GAP_003",900\n',
        encoding="utf-8",
    )

    aggregator = Aggregator()
    aggregator(str(in_dir), str(out_file), SourceDatasets.sme.value)

    df = pd.read_csv(out_file, dtype=str)

    # TIN with gap: 2 rows (island 1: 2019-02-10 to 2019-05-10, island 2: 2020-11-10 to 2020-11-10)
    gap_tin = df.loc[df["tin"] == "9999999999"]
    assert len(gap_tin) == 2, "TIN with gap should produce 2 separate islands"
    assert list(gap_tin["start_date"]) == ["2019-02-10", "2020-11-10"]
    assert list(gap_tin["end_date"]) == ["2019-05-10", "2020-11-10"]
    assert list(gap_tin["org_short_name"]) == ["ООО GAP", "ООО GAP"]

    # TIN without gap: 1 row (all three dates consecutive)
    no_gap_tin = df.loc[df["tin"] == "7777777777"]
    assert len(no_gap_tin) == 1, "TIN without gap should produce 1 merged row"
    assert no_gap_tin["start_date"].iloc[0] == "2019-02-10"
    assert no_gap_tin["end_date"].iloc[0] == "2020-11-10"


def test_aggregate_sme_no_gap_when_date_absent(tmp_path):
    """Test that when a date is absent from the table entirely, it is NOT a gap.
    TIN with 2019-02-10 and 2020-11-10 should produce 1 row if 2019-05-10
    does not exist for any TIN in the table.
    Each input file must have exactly one data_date (per-file validation).
    """
    in_dir = tmp_path / "sme_no_gap"
    in_dir.mkdir()
    out_file = tmp_path / "aggregated.csv"

    header = (
        '"kind","category","reestr_date","data_date","ind_tin","ind_number","first_name",'
        '"last_name","patronymic","org_name","org_short_name","org_tin","org_number",'
        '"region_code","region_name","region_type","district_name","district_type",'
        '"city_name","city_type","settlement_name","settlement_type","activity_code_main",'
        '"file_id","doc_cnt"\n'
    )
    # Only 2 dates in table: 2019-02-10 and 2020-11-10 (no 2019-05-10)
    # TIN 6666666666 has both dates -> no gap, 1 row. One file per date.
    (in_dir / "data-sme-no-gap-201902.csv").write_text(
        header
        + '1,1,"10.12.2018","10.02.2019",,,,,,"ООО ""TWO DATES""","ООО TD",6666666666,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_ABS_001",900\n',
        encoding="utf-8",
    )
    (in_dir / "data-sme-no-gap-202011.csv").write_text(
        header
        + '1,1,"10.12.2018","10.11.2020",,,,,,"ООО ""TWO DATES""","ООО TD",6666666666,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_ABS_001",900\n',
        encoding="utf-8",
    )

    aggregator = Aggregator()
    aggregator(str(in_dir), str(out_file), SourceDatasets.sme.value)

    df = pd.read_csv(out_file, dtype=str)

    # No gap: 2019-05-10 absent from table -> 1 merged row
    tin = df.loc[df["tin"] == "6666666666"]
    assert len(tin) == 1, "Date absent from table should not create a gap"
    assert tin["start_date"].iloc[0] == "2019-02-10"
    assert tin["end_date"].iloc[0] == "2020-11-10"


def test_aggregate_sme_data_quality_error_multiple_dates(tmp_path):
    """When a file has multiple dates with freq >= 1%, raise RuntimeError."""
    in_dir = tmp_path / "sme_bad"
    in_dir.mkdir()
    out_file = tmp_path / "aggregated.csv"

    header = (
        '"kind","category","reestr_date","data_date","ind_tin","ind_number","first_name",'
        '"last_name","patronymic","org_name","org_short_name","org_tin","org_number",'
        '"region_code","region_name","region_type","district_name","district_type",'
        '"city_name","city_type","settlement_name","settlement_type","activity_code_main",'
        '"file_id","doc_cnt"\n'
    )
    # One file with 2 dates, ~50% each -> cannot correct
    (in_dir / "data-sme-bad.csv").write_text(
        header
        + '1,1,"10.12.2018","10.02.2019",,,,,,"ООО A","ООО A",1111111111,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_001",900\n'
        + '1,1,"10.12.2018","10.05.2019",,,,,,"ООО B","ООО B",2222222222,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_001",900\n',
        encoding="utf-8",
    )

    aggregator = Aggregator()
    with pytest.raises(RuntimeError, match="Data quality error.*multiple"):
        aggregator(str(in_dir), str(out_file), SourceDatasets.sme.value)


def test_aggregate_sme_data_quality_correct_low_freq(tmp_path):
    """When a file has multiple dates and one has freq < 1%, replace with majority."""
    in_dir = tmp_path / "sme_correct"
    in_dir.mkdir()
    out_file = tmp_path / "aggregated.csv"

    header = (
        '"kind","category","reestr_date","data_date","ind_tin","ind_number","first_name",'
        '"last_name","patronymic","org_name","org_short_name","org_tin","org_number",'
        '"region_code","region_name","region_type","district_name","district_type",'
        '"city_name","city_type","settlement_name","settlement_type","activity_code_main",'
        '"file_id","doc_cnt"\n'
    )
    # One file: 199 rows with 2019-02-10, 1 row with 2019-05-10 (wrong, 0.5% < 1%) -> correct
    rows_201902 = "\n".join(
        '1,1,"10.12.2018","10.02.2019",,,,,,"ООО X","ООО X",{:010d},,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_X",900'.format(i)
        for i in range(100, 299)
    )
    row_wrong = (
        '1,1,"10.12.2018","10.05.2019",,,,,,"ООО X","ООО X",9999999999,,52,'
        '"НИЖЕГОРОДСКАЯ","ОБЛАСТЬ",,,"НИЖНИЙ НОВГОРОД","ГОРОД",,,"47.11","VO_X",900'
    )
    (in_dir / "data-sme-correct.csv").write_text(
        header + rows_201902 + "\n" + row_wrong + "\n",
        encoding="utf-8",
    )

    aggregator = Aggregator()
    aggregator(str(in_dir), str(out_file), SourceDatasets.sme.value)

    df = pd.read_csv(out_file, dtype=str)
    # All 200 TINs should have data_date 2019-02-10 after correction
    assert len(df) == 200
    assert df["start_date"].nunique() == 1
    assert df["start_date"].iloc[0] == "2019-02-10"


def test_aggregate_revexp(tmp_path):
    in_dir = pathlib.Path(__file__).parent / "data/revexp"
    out_file = tmp_path / "aggregated.csv"

    aggregator = Aggregator()

    aggregator(str(in_dir), str(out_file), SourceDatasets.revexp.value)

    df = pd.read_csv(out_file, dtype=str)

    assert len(df) == 29

    sample_record = df.loc[df["tin"] == "7727341398"]
    assert len(sample_record) == 2
    assert list(sample_record["revenue"]) == ["8831000.0", "8831000.0"]
    assert list(sample_record["expenditure"]) == ["7977000.0", "7977000.0"]
    assert list(sample_record["year"]) == ["2018", "2019"]

    correction_record = df.loc[df["tin"] == "9701103160"]
    assert len(correction_record) == 2
    assert list(correction_record["revenue"]) == ["25000.0", "24000.0"]
    assert list(correction_record["expenditure"]) == ["10000.0", "15000.0"]
    assert list(correction_record["year"]) == ["2018", "2019"]

    one_year_record = df.loc[df["tin"] == "3128132911"]
    assert len(one_year_record) == 1
    assert list(one_year_record["revenue"]) == ["1.245E7"]
    assert list(one_year_record["expenditure"]) == ["1.2367E7"]
    assert list(one_year_record["year"]) == ["2019"]


def test_aggregate_empl(tmp_path):
    in_dir = pathlib.Path(__file__).parent / "data/empl"
    out_file = tmp_path / "aggregated.csv"

    aggregator = Aggregator()

    aggregator(str(in_dir), str(out_file), SourceDatasets.empl.value)

    df = pd.read_csv(out_file, dtype=str)

    assert len(df) == 19

    sample_record = df.loc[df["tin"] == "8605025703"]
    assert len(sample_record) == 2
    assert list(sample_record["employees_count"]) == ["1", "1"]
    assert list(sample_record["year"]) == ["2018", "2019"]

    correction_record = df.loc[df["tin"] == "7801635220"]
    assert len(correction_record) == 2
    assert list(correction_record["employees_count"]) == ["10", "1"]
    assert list(correction_record["year"]) == ["2018", "2019"]

    one_year_record = df.loc[df["tin"] == "7839500199"]
    assert len(one_year_record) == 1
    assert list(one_year_record["employees_count"]) == ["30"]
    assert list(one_year_record["year"]) == ["2019"]


def test_aggregate_revexp_empl_filter_by_tin(tmp_path):
    in_dir_sme = pathlib.Path(__file__).parent / "data/sme"
    out_file_sme = tmp_path / "aggregated_sme.csv"

    in_dir_revexp = pathlib.Path(__file__).parent / "data/revexp"
    out_file_revexp = tmp_path / "aggregated_revexp.csv"

    in_dir_empl = pathlib.Path(__file__).parent / "data/empl"
    out_file_empl = tmp_path / "aggregated_empl.csv"

    aggregator = Aggregator()

    aggregator(str(in_dir_sme), str(out_file_sme), SourceDatasets.sme.value)
    aggregator(str(in_dir_revexp), str(out_file_revexp), SourceDatasets.revexp.value, str(out_file_sme))
    aggregator(str(in_dir_empl), str(out_file_empl), SourceDatasets.empl.value, str(out_file_sme))

    revexp_df = pd.read_csv(out_file_revexp, dtype=str)
    assert len(revexp_df) == 0

    empl_df = pd.read_csv(out_file_empl, dtype=str)
    assert list(empl_df["tin"].unique()) == ["8901037297"]
