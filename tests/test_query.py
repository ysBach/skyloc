"""Tests for skyloc.query convenience helpers."""

import json
import pandas as pd
import pytest

import skyloc.query as query


def test_select_orb_by_desig_matches_manual_filter():
    """select_orb_by_desig should match manual isin + reset_index."""
    orb = pd.DataFrame(
        {
            "desig": ["1 Ceres", "2 Pallas", "4 Vesta", "10 Hygiea"],
            "ecc": [0.1, 0.2, 0.3, 0.4],
        }
    )
    want = orb.loc[orb["desig"].isin(["4 Vesta", "1 Ceres"])].reset_index(drop=True)
    got = query.select_orb_by_desig(orb, ["4 Vesta", "1 Ceres"])
    pd.testing.assert_frame_equal(got, want)


def test_select_orb_by_desig_accepts_single_string():
    """Single-string input should be treated as one designation."""
    orb = pd.DataFrame({"desig": ["A", "B"], "ecc": [0.01, 0.02]})
    got = query.select_orb_by_desig(orb, "B")
    assert got["desig"].tolist() == ["B"]
    assert got.index.tolist() == [0]


def test_select_orb_by_desig_rejects_empty():
    """Empty designation inputs should fail fast."""
    orb = pd.DataFrame({"desig": ["A"], "ecc": [0.01]})
    with pytest.raises(ValueError):
        query.select_orb_by_desig(orb, [])


def test_fetch_orb_by_desig_passes_filters_and_resets(monkeypatch):
    """fetch_orb_by_desig should apply desig filter and reset indices."""
    captured = {}

    def _fake_fetch_orb(**kwargs):
        captured.update(kwargs)
        orb = pd.DataFrame(
            {"desig": ["1 Ceres", "4 Vesta"], "ecc": [0.1, 0.3]},
            index=[10, 20],
        )
        m_ng = pd.Series([False, True], index=[10, 20], name="m_ng")
        return orb, m_ng

    monkeypatch.setattr(query, "fetch_orb", _fake_fetch_orb)

    got_orb, got_m_ng = query.fetch_orb_by_desig(["1 Ceres", "4 Vesta"])

    assert captured["filters"] == [("desig", "in", ("1 Ceres", "4 Vesta"))]
    assert got_orb.index.tolist() == [0, 1]
    assert got_m_ng.index.tolist() == [0, 1]
    assert got_orb["desig"].tolist() == ["1 Ceres", "4 Vesta"]


def test_query_orb_by_desig_queries_only_requested_designations(monkeypatch):
    """query_orb_by_desig should query each requested desig and preserve order."""
    captured_sb_cdata = []

    class FakeSBDBQuery:
        def __init__(self, **kwargs):
            self.sb_cdata = kwargs["sb_cdata"]
            captured_sb_cdata.append(self.sb_cdata)

        def query(self, **kwargs):
            desig = json.loads(self.sb_cdata)["AND"][0].split("|")[-1]
            if desig == "missing":
                return pd.DataFrame(columns=["desig", "ecc"])
            return pd.DataFrame({"desig": [desig], "ecc": [0.1]})

    monkeypatch.setattr(query, "SBDBQuery", FakeSBDBQuery)

    got = query.query_orb_by_desig(["1 Ceres", "4 Vesta"])
    assert len(captured_sb_cdata) == 2
    assert got["desig"].tolist() == ["1 Ceres", "4 Vesta"]


def test_query_orb_by_desig_strict_raises_on_missing(monkeypatch):
    """strict=True should raise if any requested desig is not found."""

    class FakeSBDBQuery:
        def __init__(self, **kwargs):
            self.sb_cdata = kwargs["sb_cdata"]

        def query(self, **kwargs):
            if "missing" in self.sb_cdata:
                return pd.DataFrame(columns=["desig", "ecc"])
            return pd.DataFrame({"desig": ["1 Ceres"], "ecc": [0.1]})

    monkeypatch.setattr(query, "SBDBQuery", FakeSBDBQuery)

    with pytest.raises(ValueError, match="Could not find"):
        query.query_orb_by_desig(["1 Ceres", "missing"], strict=True)
