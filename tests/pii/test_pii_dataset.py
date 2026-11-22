from benchmarks_path import evaluate  # noqa: I001
from pii_dataset import generate


def test_dataset_size_and_quality() -> None:
    samples = generate()
    assert len(samples) >= 300
    res = evaluate(samples)
    for entity, m in res.items():
        assert m["support"] > 0, entity
        assert m["precision"] >= 0.98, (entity, m)
        assert m["recall"] >= 0.95, (entity, m)
