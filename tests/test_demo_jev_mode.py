"""The Jev-mode demo on the synthetic model: every section runs, every planted effect shows, no optional deps."""
import json
import sys

import pytest

from demo import jev_mode


def test_fake_walkthrough_end_to_end(tmp_path, capsys):
    out = tmp_path / "demo.json"
    jev_mode.main(["--backend", "fake", "--json", str(out), "--no-color"])
    r = json.loads(out.read_text())
    lv = r["levels"]["route"]
    assert lv["raw"]["acc"] < lv["L0"]["acc"] < lv["L2"]["acc"] and lv["L2"]["acc"] >= 0.9
    assert lv["L2"]["blocks_executed"] == 3 and lv["L2"]["n_blocks"] == 4
    assert lv["L0"]["prompts_per_decision"] == 4 and lv["L2"]["prompts_per_decision"] == 1
    ro = r["routing"]["route"]
    assert ro["as_is"]["acc"] <= ro["recentred"]["acc"] - 0.1 and ro["recentred"]["acc"] >= 0.9
    assert ro["recentred"]["adapt_n"] == 100 and ro["routed_from"] == "route"   # 50 states, served twice
    assert ro["reversed"]["reordered"] is True and ro["reversed"]["acc"] >= 0.9
    # the artifact stores float32; on the planted world the probabilities saturate, so |dp| is far below this
    assert r["roundtrip"]["max_abs_dp"] < 1e-6 and r["roundtrip"]["n_heads"] == 1
    assert r["roundtrip"]["auto_without_head"] == "L0"
    text = capsys.readouterr().out
    assert "[1/4]" in text and "routed" in text and "L2" in text


def test_own_state_and_question_on_the_fake(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({"ticket": "charged twice"}))
    q = '{"kind":"choice","text":"Which handler?","options":["billing","technical","sales","other"]}'
    out = tmp_path / "one.json"
    jev_mode.main(["--backend", "fake", "--state-file", str(tmp_path / "state.json"), "--question", q,
                   "--json", str(out)])
    d = json.loads(out.read_text())["decision"]
    assert d["level"] == "L0" and abs(sum(d["distribution"].values()) - 1) < 1e-9
    labels = [{"state": f"ticket {i}", "label": i % 4} for i in range(16)]
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    jev_mode.main(["--backend", "fake", "--state-file", str(tmp_path / "state.json"), "--question", q,
                   "--labels-file", str(tmp_path / "labels.json"), "--save-head", str(tmp_path / "head.json"),
                   "--json", str(out)])
    d = json.loads(out.read_text())["decision"]
    assert d["level"] == "L2" and json.loads((tmp_path / "head.json").read_text())["heads"]


def test_missing_datasets_is_a_clean_error_before_any_model_load(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)      # import raises ImportError
    monkeypatch.setitem(sys.modules, "torch", None)          # must not be reached
    with pytest.raises(SystemExit) as e:
        jev_mode.main(["--backend", "hf"])
    assert "datasets" in str(e.value) and "pip install" in str(e.value)


def test_question_specs_and_names():
    assert jev_mode.parse_question_spec('{"kind":"noul","text":"Is it?"}').kind == "noul"
    assert jev_mode.parse_question_spec('{"kind":"score","text":"How bad?","levels":["low","high"]}').ordered
    assert len(jev_mode.TYPED_QUESTION_NAMES) == 20
    assert jev_mode.resolve_question_names("typed.customer_service.category,invoice_processing") == [
        "customer_service.category"] + [n for n in jev_mode.TYPED_QUESTION_NAMES if n.startswith("invoice_processing.")]
    with pytest.raises(SystemExit):
        jev_mode.resolve_question_names("customer_serv.category")


def test_lifecycle_on_the_fake_grows_the_head_from_traffic_and_survives_a_restart(tmp_path):
    out = tmp_path / "life.json"
    jev_mode.main(["--backend", "fake", "--lifecycle", "--json", str(out), "--no-color"])
    r = json.loads(out.read_text())["lifecycle"]
    assert r["day0"]["level"] == "L0"
    assert [f["at"] for f in r["fits"]] == [30, 60, 120] and all(f["level"] == "L2" for f in r["fits"])
    assert r["fits"][0]["acc"] > r["day0"]["acc"] and r["fits"][-1]["acc"] >= 0.9
    w = r["wording"]
    assert w["routed_from"] == "route" and w["acc_after_adapted"] >= 0.9
    assert w["acc_after_adapted"] >= w["acc_after_control"] + 0.1
    assert r["restart"]["adapted_on_first_request"] and r["restart"]["max_abs_dp"] < 0.05
    assert r["restart"]["n_observations"] == 150 and r["restart"]["n_heads"] == 1
