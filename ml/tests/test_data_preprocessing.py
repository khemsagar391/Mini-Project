import pandas as pd
import pytest

from data_preprocessing import LABEL_MAP, load_cicids_file, time_split

HEADER = (
    "Flow ID, Source IP, Timestamp, Flow Duration, Total Fwd Packets, "
    "Total Backward Packets, Total Length of Fwd Packets, "
    "Total Length of Bwd Packets, Label\n"
)


def write(tmp_path, body, name="f.csv"):
    p = tmp_path / name
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def test_padding_rows_are_dropped(tmp_path):
    p = write(tmp_path,
              "a,1.1.1.1,06/07/2017 12:59,10,1,1,5,5,BENIGN\n"
              ",,,,,,,,\n")
    df = load_cicids_file(p)
    assert len(df) == 1


def test_web_attack_labels_are_mapped(tmp_path):
    bad = "Web Attack \ufffd XSS"
    p = write(tmp_path, f"a,1.1.1.1,06/07/2017 12:59,10,1,1,5,5,{bad}\n")
    df = load_cicids_file(p)
    assert df["label"].iloc[0] == LABEL_MAP[bad] == "Web Attack - XSS"


def test_both_timestamp_formats_parse(tmp_path):
    p = write(tmp_path,
              "a,1.1.1.1,06/07/2017 12:59,10,1,1,5,5,BENIGN\n"
              "a,1.1.1.1,03/07/2017 08:55:58,10,1,1,5,5,BENIGN\n")
    df = load_cicids_file(p)
    assert df["timestamp"].iloc[0] == pd.Timestamp("2017-07-06 12:59")
    assert df["timestamp"].iloc[1] == pd.Timestamp("2017-07-03 08:55:58")


def test_bad_timestamp_raises(tmp_path):
    p = write(tmp_path, "a,1.1.1.1,not-a-date,10,1,1,5,5,BENIGN\n")
    with pytest.raises(RuntimeError, match="unparsed timestamps"):
        load_cicids_file(p)


def test_non_numeric_duration_raises(tmp_path):
    p = write(tmp_path, "a,1.1.1.1,06/07/2017 12:59,abc,1,1,5,5,BENIGN\n")
    with pytest.raises(RuntimeError, match="non-numeric"):
        load_cicids_file(p)


def test_time_split_is_per_label_and_ordered(tmp_path):
    rows = []
    for i in range(10):
        rows.append(f"a,1.1.1.1,06/07/2017 {i:02d}:00,10,1,1,5,5,PortScan\n")
    for i in range(10):
        rows.append(f"a,1.1.1.1,06/07/2017 {i:02d}:30,10,1,1,5,5,BENIGN\n")
    p = write(tmp_path, "".join(rows))
    df = time_split(load_cicids_file(p))

    scan = df[df["label"] == "PortScan"]
    counts = scan["split"].value_counts()
    assert counts["train"] == 7 and counts["val"] == 1 and counts["test"] == 2

    # Within the label, every train row is earlier than every test row.
    assert scan.loc[scan["split"] == "train", "timestamp"].max() < \
        scan.loc[scan["split"] == "test", "timestamp"].min()

    # Every label has training rows.
    for label in df["label"].unique():
        assert "train" in set(df.loc[df["label"] == label, "split"])


def test_time_split_rejects_tiny_label(tmp_path):
    p = write(tmp_path, "a,1.1.1.1,06/07/2017 12:59,10,1,1,5,5,BENIGN\n")
    with pytest.raises(RuntimeError, match="need at least"):
        time_split(load_cicids_file(p))