from __future__ import annotations

import numpy as np
import pytest

from kinematic_classifier_sandbox.corpus.real_world.adapters.space_orbital_sp3 import (
    parse_sp3_text,
    records_to_si_arrays as sp3_records_to_si_arrays,
    select_satellite_records,
)


SP3_TEXT = """#cP2024  1  3  0  0  0.00000000       2 ORBIT IGS20 HLM  IGS
## 2295 259200.00000000   900.00000000 60312 0.0000000000000
+    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
/* Bounded parser contract fixture; values are synthetic.
*  2024  1  3  0  0  0.00000000
PG01  13012.000000  -8828.000000  20993.000000    165.000000
PG02  15300.00000  -1413.000000  22180.000000   -506.000000
*  2024  1  3   0 15  0.00000000
PG01  13381.000000  -6341.000000  21668.000000    165.100000
PG02  15829.000000    934.000000  21843.000000   -506.100000
EOF
"""


def test_parse_sp3_header_and_select_satellite() -> None:
    extract = parse_sp3_text(SP3_TEXT)

    assert extract.header.version == "c"
    assert extract.header.data_type == "P"
    assert extract.header.declared_epoch_count == 2
    assert extract.header.coordinate_system == "IGS20"
    assert extract.header.orbit_type == "HLM"
    assert extract.header.agency == "IGS"
    assert extract.header.time_system == "GPS"
    assert extract.header.sampling_period_s == 900.0
    assert extract.header.declared_satellite_count == 2
    assert extract.header.satellite_ids == ("G01", "G02")

    records = select_satellite_records(extract, "g01")
    assert len(records) == 2
    timestamps_s, position_m, clock_offset_s = sp3_records_to_si_arrays(
        extract.header,
        records,
    )
    assert timestamps_s[0] == pytest.approx(1_388_275_200.0)
    assert np.diff(timestamps_s) == pytest.approx(np.asarray([900.0]))
    assert position_m[0] == pytest.approx((13_012_000.0, -8_828_000.0, 20_993_000.0))
    assert clock_offset_s == pytest.approx(np.asarray([165.0e-6, 165.1e-6]))
####


def test_sp3_parser_rejects_missing_position_vector() -> None:
    corrupted = SP3_TEXT.replace(
        "13012.000000  -8828.000000  20993.000000",
        "    0.000000      0.000000      0.000000",
        1,
    )

    with pytest.raises(ValueError, match="position missing"):
        parse_sp3_text(corrupted)
####


def test_sp3_parser_rejects_nonstandard_position_sentinel() -> None:
    corrupted = SP3_TEXT.replace("13012.000000", "999999.999999", 1)

    with pytest.raises(ValueError, match="position missing"):
        parse_sp3_text(corrupted)
####


def test_sp3_parser_marks_missing_clock_as_nan() -> None:
    source = SP3_TEXT.replace("165.000000", "999999.000000", 1)
    extract = parse_sp3_text(source)
    records = select_satellite_records(extract, "G01")
    _, _, clock_offset_s = sp3_records_to_si_arrays(extract.header, records)

    assert np.isnan(clock_offset_s[0])
    assert clock_offset_s[1] == pytest.approx(165.1e-6)
####


def test_sp3_selection_rejects_unknown_satellite() -> None:
    extract = parse_sp3_text(SP3_TEXT)

    with pytest.raises(ValueError, match="at least two records"):
        select_satellite_records(extract, "G99")
####


def test_sp3_parser_rejects_declared_epoch_count_mismatch() -> None:
    corrupted = SP3_TEXT.replace("       2 ORBIT", "       3 ORBIT", 1)

    with pytest.raises(ValueError, match="declared epoch count"):
        parse_sp3_text(corrupted)
####


def test_sp3_parser_rejects_start_epoch_mismatch() -> None:
    corrupted = SP3_TEXT.replace(
        "*  2024  1  3  0  0  0.00000000",
        "*  2024  1  3  0  1  0.00000000",
        1,
    )

    with pytest.raises(ValueError, match="first epoch"):
        parse_sp3_text(corrupted)
####


def test_sp3_parser_rejects_cadence_mismatch() -> None:
    corrupted = SP3_TEXT.replace(
        "*  2024  1  3  0 15  0.00000000",
        "*  2024  1  3  0 16  0.00000000",
        1,
    )

    with pytest.raises(ValueError, match="cadence"):
        parse_sp3_text(corrupted)
####


def test_sp3_parser_rejects_duplicate_satellite_record() -> None:
    corrupted = SP3_TEXT.replace(
        "PG02  15300.000000  -1413.000000  22180.000000   -506.000000",
        "PG01  15300.000000  -1413.000000  22180.000000   -506.000000",
        1,
    )

    with pytest.raises(ValueError, match="duplicate SP3 position record"):
        parse_sp3_text(corrupted)
####


def test_sp3_parser_rejects_epoch_roster_mismatch() -> None:
    corrupted = SP3_TEXT.replace(
        "PG02  15829.000000    934.000000  21843.000000   -506.100000\n",
        "",
        1,
    )

    with pytest.raises(ValueError, match="record roster"):
        parse_sp3_text(corrupted)
####


def test_sp3_si_conversion_rejects_non_gps_time_system() -> None:
    source = SP3_TEXT.replace("%c G  cc GPS", "%c G  cc UTC", 1)
    extract = parse_sp3_text(source)
    records = select_satellite_records(extract, "G01")

    with pytest.raises(ValueError, match="requires an SP3 header declaring time system GPS"):
        sp3_records_to_si_arrays(extract.header, records)
####


def test_sp3_si_conversion_rejects_mixed_satellites() -> None:
    extract = parse_sp3_text(SP3_TEXT)
    mixed = (extract.records[0], extract.records[1], extract.records[2])

    with pytest.raises(ValueError, match="one satellite"):
        sp3_records_to_si_arrays(extract.header, mixed)
####
