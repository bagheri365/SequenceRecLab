from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs" / "datasets.toml"


def load_registry() -> dict:
    with REGISTRY_PATH.open("rb") as handle:
        return tomllib.load(handle)


def test_registry_contains_required_datasets() -> None:
    registry = load_registry()
    datasets = registry["datasets"]

    assert {"yoochoose", "retailrocket", "movielens_1m"} <= set(datasets)


def test_every_dataset_has_semantic_contract() -> None:
    registry = load_registry()
    required_fields = {
        "role",
        "domain",
        "event_unit",
        "timestamp_semantics",
        "identity_scope",
        "sequence_semantics",
        "repeat_items_meaningful",
        "seen_item_policy",
        "primary_target_event",
        "recommended_history_lengths",
        "source_url",
        "notes",
    }

    for name, config in registry["datasets"].items():
        missing = required_fields - set(config)
        assert not missing, f"{name} missing fields: {sorted(missing)}"
        assert config["sequence_semantics"] in {"strong", "weak"}
        assert config["seen_item_policy"] in {"allow", "mask"}
        assert config["recommended_history_lengths"] == sorted(
            set(config["recommended_history_lengths"])
        )
        assert all(length >= 1 for length in config["recommended_history_lengths"])


def test_primary_dataset_has_strong_sequence_semantics() -> None:
    registry = load_registry()
    primary = [
        (name, config)
        for name, config in registry["datasets"].items()
        if config["role"] == "primary"
    ]

    assert primary, "at least one primary dataset is required"
    for name, config in primary:
        assert config["sequence_semantics"] == "strong", (
            f"primary dataset {name} must have strong sequence semantics"
        )


def test_weak_sequence_dataset_cannot_be_primary() -> None:
    registry = load_registry()

    for name, config in registry["datasets"].items():
        if config["sequence_semantics"] == "weak":
            assert config["role"] != "primary", (
                f"weak-order dataset {name} cannot carry the primary behavioral claim"
            )


def test_repeat_policy_is_domain_specific() -> None:
    registry = load_registry()["datasets"]

    assert registry["yoochoose"]["seen_item_policy"] == "allow"
    assert registry["retailrocket"]["seen_item_policy"] == "allow"
    assert registry["movielens_1m"]["seen_item_policy"] == "mask"


def test_yoochoose_real_data_history_protocol():
    config = load_registry()["datasets"]["yoochoose"]
    assert config["recommended_history_lengths"] == [2, 3, 5]
    assert config["secondary_history_lengths"] == [10]
