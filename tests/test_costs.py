from decimal import Decimal

from gateway.costs import compute_cost, is_priced
from gateway.schemas import Usage


def test_cost_computation():
    assert compute_cost(
        "claude-sonnet-5-5", Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    ) == Decimal(18)
    assert compute_cost("llama3.2", Usage(prompt_tokens=999, completion_tokens=999)) == 0


def test_unknown_model_is_free_but_flagged():
    assert not is_priced("mystery")
    assert compute_cost("mystery", Usage(prompt_tokens=10)) == 0
