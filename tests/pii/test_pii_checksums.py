import random

from hypothesis import given
from hypothesis import strategies as st
from pii_dataset import make_bin, make_card, make_iban, make_iin

from gateway.middleware.pii import (
    iban_valid,
    iin_check_digit,
    is_valid_bin,
    is_valid_iin,
    luhn_valid,
)
from gateway.middleware.pii.detectors import iin_checksum_valid

seeds = st.integers(min_value=0, max_value=2**32)


def test_known_values() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert iban_valid("KZ86125KZT5004100100")
    assert iban_valid("KZ86 125K ZT50 0410 0100")
    assert not iban_valid("KZ87125KZT5004100100")


@given(seeds)
def test_generated_iin_valid(seed: int) -> None:
    assert is_valid_iin(make_iin(random.Random(seed)))


@given(seeds)
def test_generated_bin_valid(seed: int) -> None:
    v = make_bin(random.Random(seed))
    assert is_valid_bin(v) and not is_valid_iin(v)


@given(seeds, st.integers(min_value=1, max_value=9))
def test_iin_wrong_check_digit_rejected(seed: int, delta: int) -> None:
    v = make_iin(random.Random(seed))
    bad = v[:11] + str((int(v[11]) + delta) % 10)
    assert not iin_checksum_valid(bad)


@given(st.text(alphabet="0123456789", min_size=11, max_size=11))
def test_iin_check_digit_roundtrip(body: str) -> None:
    c = iin_check_digit(body)
    if c is not None:
        assert iin_checksum_valid(body + str(c))
    else:
        assert not any(iin_checksum_valid(body + str(d)) for d in range(10))


@given(seeds)
def test_generated_card_luhn(seed: int) -> None:
    assert luhn_valid(make_card(random.Random(seed)))


@given(seeds, st.integers(min_value=1, max_value=9), st.integers(min_value=0, max_value=15))
def test_card_single_digit_change_detected(seed: int, delta: int, pos: int) -> None:
    c = make_card(random.Random(seed))
    bad = c[:pos] + str((int(c[pos]) + delta) % 10) + c[pos + 1 :]
    assert not luhn_valid(bad)  # Luhn catches every single-digit error


@given(seeds)
def test_generated_iban_valid(seed: int) -> None:
    assert iban_valid(make_iban(random.Random(seed)))


@given(seeds, st.integers(min_value=1, max_value=98))
def test_iban_wrong_check_rejected(seed: int, delta: int) -> None:
    v = make_iban(random.Random(seed))
    bad = f"KZ{(int(v[2:4]) + delta) % 100:02d}" + v[4:]
    assert not iban_valid(bad)
