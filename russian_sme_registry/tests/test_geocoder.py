import pathlib

import numpy as np
import pandas as pd

from ..stages.geocode import LocalGeocoder


def test_geocode(tmp_path):
    in_file = pathlib.Path(__file__).parent / "data/sme/test-aggregated.csv"
    out_file = tmp_path / "geocoded.csv"

    geocoder = LocalGeocoder()

    geocoder(str(in_file), str(out_file))

    df = pd.read_csv(out_file, dtype=str)

    assert len(df) == 37

    expected_cols = [
        "tin",
        "reg_number",
        "kind",
        "category",
        "first_name",
        "last_name",
        "patronymic",
        "org_name",
        "org_short_name",
        "activity_code_main",
        "region_iso_code",
        "region_code",
        "region",
        "area",
        "settlement",
        "settlement_type",
        "oktmo",
        "lat",
        "lon",
        "start_date",
        "end_date",
    ]

    assert list(df.columns) == expected_cols

    assert sorted(df["region_code"].unique()) == ["52", "59", "66", "77", "78", "89"]
    assert sorted(df["region_iso_code"].unique()) == ["RU-MOW", "RU-NIZ", "RU-PER", "RU-SPE", "RU-SVE", "RU-YAN"]

    ekb_record = df.loc[df["tin"] == "890104055324"]

    assert list(ekb_record["region_code"]) == ["66"]
    assert list(ekb_record["region_iso_code"]) == ["RU-SVE"]
    assert list(ekb_record["region"]) == ["Свердловская область"]
    assert list(ekb_record["area"]) == [np.nan]
    assert list(ekb_record["settlement"]) == ["Екатеринбург"]
    assert list(ekb_record["settlement_type"]) == ["г"]
    assert list(ekb_record["oktmo"]) == ["65701000001"]
    assert list(ekb_record["lat"]) == ["56.83852"]
    assert list(ekb_record["lon"]) == ["60.60549"]

    spasskoe_record = df.loc[df["tin"] == "523200013760"]

    assert list(spasskoe_record["region_code"]) == ["52"]
    assert list(spasskoe_record["region_iso_code"]) == ["RU-NIZ"]
    assert list(spasskoe_record["region"]) == ["Нижегородская область"]
    assert list(spasskoe_record["area"]) == ["Спасский район"]
    assert list(spasskoe_record["settlement"]) == ["Спасское"]
    assert list(spasskoe_record["settlement_type"]) == ["с"]
    assert list(spasskoe_record["oktmo"]) == ["22651432101"]
    assert list(spasskoe_record["lat"]) == ["55.858055"]
    assert list(spasskoe_record["lon"]) == ["45.695835"]

    moscow_record = df.loc[df["tin"] == "591604272151"]

    assert list(moscow_record["region_code"]) == ["77"]
    assert list(moscow_record["region_iso_code"]) == ["RU-MOW"]
    assert list(moscow_record["region"]) == ["Москва"]
    assert list(moscow_record["area"]) == [np.nan]
    assert list(moscow_record["settlement"]) == ["Москва"]
    assert list(moscow_record["settlement_type"]) == ["г"]
    assert list(moscow_record["oktmo"]) == ["45000000"]
    assert list(moscow_record["lat"]) == ["55.754047"]
    assert list(moscow_record["lon"]) == ["37.620403"]

    krasnokamsk_record = df.loc[df["tin"] == "5916034536"].head(1)

    assert list(krasnokamsk_record["region_code"]) == ["59"]
    assert list(krasnokamsk_record["region_iso_code"]) == ["RU-PER"]
    assert list(krasnokamsk_record["region"]) == ["Пермский край"]
    assert list(krasnokamsk_record["area"]) == [np.nan]
    assert list(krasnokamsk_record["settlement"]) == ["Краснокамск"]
    assert list(krasnokamsk_record["settlement_type"]) == ["г"]
    assert list(krasnokamsk_record["oktmo"]) == ["57720000001"]
    assert list(krasnokamsk_record["lat"]) == ["58.082207"]
    assert list(krasnokamsk_record["lon"]) == ["55.747993"]

    zavolzhye_record = df.loc[df["tin"] == "5234003743"]

    assert list(zavolzhye_record["region_code"]) == ["52"]
    assert list(zavolzhye_record["region_iso_code"]) == ["RU-NIZ"]
    assert list(zavolzhye_record["region"]) == ["Нижегородская область"]
    assert list(zavolzhye_record["area"]) == ["Городецкий район"]
    assert list(zavolzhye_record["settlement"]) == ["Заволжье"]
    assert list(zavolzhye_record["settlement_type"]) == ["г"]
    assert list(zavolzhye_record["oktmo"]) == ["22628103001"]
    assert list(zavolzhye_record["lat"]) == ["56.64043"]
    assert list(zavolzhye_record["lon"]) == ["43.38725"]
