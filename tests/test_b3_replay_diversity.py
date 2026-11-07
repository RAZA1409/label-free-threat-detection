"""
tests/test_b3_replay_diversity.py
Comprehensive regression tests for AEGIS Problem B3: Option B-1 Two-Episode Replay.

Verifies:
1. Default replay selects exactly 2,000 authentic chronological flows.
2. Episode 1 contains 1,000 flows of PortScan with 100 pure BENIGN lead-in flows.
3. Episode 2 contains 1,000 flows of DDoS onset with 100 pure BENIGN lead-in flows.
4. Concatenated stream is globally timestamp-monotonic with zero time reversal.
5. Buffer purges completely at the 3,900-second boundary in 10-second causal event-time.
6. Legacy DDoS replay remains accessible via replay_scenario='legacy_ddos'.
7. v1 feature dimension (32) and v2 shadow dimension (26) are preserved.
8. Safe handling of max_events variations (None, 0, 1, 50, 1000, 2000) and error rejection.
9. Validation of replay_scenario parameter against supported options.
10. Transparent metadata and episode_slices tracking.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from features import AUGMENTED_ONE_WAY_FEATURES, PURE_ONE_WAY_FEATURES, sanitize_labels
from streaming_features import (
    StreamingFeatureEngine,
    FlowQuarantineManager,
    prepare_chronological_stream,
    PORTSCAN_REPLAY_SLICE,
    DDOS_ONSET_REPLAY_SLICE,
    SUPPORTED_REPLAY_SCENARIOS,
)
from explainability import ThreatExplainer
from model_adapters import PureOneWayV2Adapter


@pytest.fixture(scope="module")
def full_test_dataset():
    """Loads the test dataset if present on disk; skips if absent."""
    p = ROOT / "outputs" / "experiment" / "test_tue_fri.parquet"
    if not p.exists():
        pytest.skip(f"Test parquet {p} not available in this environment.")
    df = pd.read_parquet(p)
    df["Clean_Label"] = sanitize_labels(df["Label"])
    return df


def test_01_two_episode_default_composition(full_test_dataset):
    """Verify default prepare_chronological_stream creates exactly 2,000 flows with 2 authentic episodes."""
    stream = prepare_chronological_stream(full_test_dataset, max_events=2000, inject_attacks=True)

    assert len(stream) == 2000
    assert stream.attrs.get("selection_mode") == "two_episode_composite"

    # Episode 1: PortScan [0:1000]
    ep1 = stream.iloc[:1000]
    assert len(ep1) == 1000
    # First 100 flows are pure BENIGN lead-in
    assert ep1["Clean_Label"].iloc[:100].unique().tolist() == ["BENIGN"]
    # Remainder contains PortScan attacks
    assert "PortScan" in ep1["Clean_Label"].unique()
    assert (ep1["Clean_Label"] == "PortScan").sum() == 897
    assert (ep1["Clean_Label"] == "BENIGN").sum() == 103

    # Episode 2: DDoS Onset [1000:2000]
    ep2 = stream.iloc[1000:]
    assert len(ep2) == 1000
    # First 100 flows of Episode 2 are pure BENIGN lead-in
    assert ep2["Clean_Label"].iloc[:100].unique().tolist() == ["BENIGN"]
    # Remainder contains DDoS attacks
    assert "DDoS" in ep2["Clean_Label"].unique()
    assert (ep2["Clean_Label"] == "DDoS").sum() == 460
    assert (ep2["Clean_Label"] == "BENIGN").sum() == 540


def test_02_global_chronological_monotonicity(full_test_dataset):
    """Verify that timestamps across the concatenated 2,000-flow stream are strictly monotonic."""
    stream = prepare_chronological_stream(full_test_dataset, max_events=2000, inject_attacks=True)

    assert stream["Timestamp"].is_monotonic_increasing

    # Check temporal gap between Episode 1 end and Episode 2 start
    t_ep1_end = stream["Timestamp"].iloc[999]
    t_ep2_start = stream["Timestamp"].iloc[1000]
    gap = t_ep2_start - t_ep1_end

    assert gap.total_seconds() == 3900.0  # 1 hour 5 minutes
    assert gap > pd.Timedelta(seconds=10)


def test_03_causal_buffer_boundary_eviction(full_test_dataset):
    """Verify that the 10-second sliding causal buffer evicts Episode 1 before Episode 2 begins."""
    stream = prepare_chronological_stream(full_test_dataset, max_events=2000, inject_attacks=True)

    qm = FlowQuarantineManager()
    engine = StreamingFeatureEngine(
        window_seconds=10,
        clock_mode="event_time",
        quarantine_manager=qm,
    )

    # Process Episode 1 (first 1,000 flows) in chunks of 50
    for i in range(0, 1000, 50):
        chunk = stream.iloc[i : i + 50]
        engine.context.process_batch(chunk, causal_mode="simultaneous", flush=False)

    # Flush Episode 1 trailing group
    engine.context.flush()

    # Now ingest first chunk of Episode 2 (flow index 1000 to 1050)
    ep2_first_chunk = stream.iloc[1000:1050]
    engine.context.process_batch(ep2_first_chunk, causal_mode="simultaneous", flush=False)

    # The buffer must contain ONLY flows from Episode 2 timestamp (2017-07-07 03:56:00)
    for entry in engine.context.buffer:
        ts = entry[0]
        assert ts >= pd.Timestamp("2017-07-07 03:56:00"), f"Leaked historical flow timestamp {ts} into Episode 2"


def test_04_legacy_ddos_replay_compatibility(full_test_dataset):
    """Verify that legacy contiguous DDoS replay remains accessible via replay_scenario='legacy_ddos'."""
    legacy_stream = prepare_chronological_stream(
        full_test_dataset,
        max_events=2000,
        inject_attacks=True,
        replay_scenario="legacy_ddos",
    )

    assert len(legacy_stream) == 2000
    assert legacy_stream.attrs.get("selection_mode") == "contiguous_attack_burst"
    assert legacy_stream.attrs.get("window_start_idx") == 1942302
    assert (legacy_stream["Clean_Label"] == "DDoS").sum() == 1942


def test_05_feature_dimensions_and_model_roles():
    """Verify that v1 is 32-dim and v2 is 26-dim without changes to model definitions."""
    explainer = ThreatExplainer()
    v2_adapter = PureOneWayV2Adapter()

    assert len(AUGMENTED_ONE_WAY_FEATURES) == 32
    assert len(explainer.feature_names) == 32
    assert len(PURE_ONE_WAY_FEATURES) == 26
    assert len(v2_adapter.feature_names) == 26
    assert explainer.threshold == pytest.approx(0.190885, abs=1e-5)
    assert v2_adapter.primary_threshold == pytest.approx(0.314036, abs=1e-5)


def test_06_max_events_variations_and_boundary_handling(full_test_dataset):
    """Verify max_events=None, 0, 1, 50, 1000, 2000 and invalid inputs."""
    # None returns full dataset
    s_none = prepare_chronological_stream(full_test_dataset, max_events=None)
    assert len(s_none) == len(full_test_dataset)
    assert s_none.attrs.get("selection_mode") == "full_dataset"

    # 0 returns empty dataframe with schema
    s_zero = prepare_chronological_stream(full_test_dataset, max_events=0)
    assert len(s_zero) == 0
    assert s_zero.attrs.get("selection_mode") == "empty"

    # 1 returns exactly 1 flow without IndexError
    s_one = prepare_chronological_stream(full_test_dataset, max_events=1, inject_attacks=True)
    assert len(s_one) == 1
    assert s_one.attrs.get("selection_mode") == "two_episode_composite"

    # 50 returns exactly 50 flows
    s_50 = prepare_chronological_stream(full_test_dataset, max_events=50, inject_attacks=True)
    assert len(s_50) == 50

    # 1000 returns exactly 1,000 flows
    s_1000 = prepare_chronological_stream(full_test_dataset, max_events=1000, inject_attacks=True)
    assert len(s_1000) == 1000

    # 2000 returns exactly 2,000 flows
    s_2000 = prepare_chronological_stream(full_test_dataset, max_events=2000, inject_attacks=True)
    assert len(s_2000) == 2000

    # Negative max_events raises ValueError
    with pytest.raises(ValueError, match="max_events cannot be negative"):
        prepare_chronological_stream(full_test_dataset, max_events=-5)

    # Non-integer max_events raises TypeError
    with pytest.raises(TypeError, match="max_events must be an integer or None"):
        prepare_chronological_stream(full_test_dataset, max_events="2000")


def test_07_replay_scenario_validation(full_test_dataset):
    """Verify that replay_scenario validates allowed scenarios and rejects unsupported values."""
    for valid in ["two_episode", "legacy_ddos", "contiguous_burst"]:
        res = prepare_chronological_stream(full_test_dataset, max_events=50, inject_attacks=True, replay_scenario=valid)
        assert len(res) == 50

    with pytest.raises(ValueError, match="Unsupported replay_scenario"):
        prepare_chronological_stream(full_test_dataset, max_events=50, replay_scenario="invalid_foo")


def test_08_episode_slices_metadata(full_test_dataset):
    """Verify that episode_slices metadata accurately details the two discrete episodes."""
    stream = prepare_chronological_stream(full_test_dataset, max_events=2000, inject_attacks=True)
    episodes = stream.attrs.get("episode_slices")

    assert episodes is not None
    assert len(episodes) == 2
    assert episodes[0]["attack"] == "PortScan"
    assert episodes[0]["start_idx"] == PORTSCAN_REPLAY_SLICE[0]
    assert episodes[0]["count"] == 1000

    assert episodes[1]["attack"] == "DDoS"
    assert episodes[1]["start_idx"] == DDOS_ONSET_REPLAY_SLICE[0]
    assert episodes[1]["count"] == 1000
