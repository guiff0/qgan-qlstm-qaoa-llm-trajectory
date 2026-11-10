import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import check_prereg_v2 as c


def test_detects_unfilled_and_passes_when_filled(tmp_path):
    p = tmp_path / "p.md"
    p.write_text("seeds [FILL: 53-72]\nok\nrho [FILL: 0.7] and [FILL: 4]\n", encoding="utf8")
    assert [x[0] for x in c.unfilled(p.read_text())] == [1, 3, 3]
    assert c.main(str(p)) == 1
    p.write_text("seeds 53-72\n", encoding="utf8")
    assert c.main(str(p)) == 0


def test_the_shipped_draft_is_not_frozen_yet_and_lists_its_open_decisions():
    text = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PREREGISTRATION_V2.md"), encoding="utf8").read()
    open_items = c.unfilled(text)
    assert len(open_items) == 1 and "calibrate_poisoning" in open_items[0][1]       # only the data-dependent shift_sd is left
    assert "NOT_DECIDABLE" in text and "last-layer RZZ skipped" in text and "NO_FORECASTING_SIGNAL" in text
