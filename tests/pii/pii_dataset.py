"""Synthetic labeled PII dataset (RU/KZ/EN). All values are generated; no real people's data.

Checksum generators here are written independently from the detectors under test.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    text: str
    spans: tuple[tuple[int, int, str], ...]  # (start, end, entity)
    lang: str


# ------------------------------------------------------------------ value generators


def _iin_body_check(body: str) -> int | None:
    w1 = list(range(1, 12))
    w2 = [(i + 3) % 11 or 11 for i in range(11)]  # 3,4,...,11,1,2
    d = [int(c) for c in body]
    r = sum(a * b for a, b in zip(d, w1, strict=True)) % 11
    if r == 10:
        r = sum(a * b for a, b in zip(d, w2, strict=True)) % 11
    return None if r == 10 else r


def make_iin(rng: random.Random) -> str:
    while True:
        yy, mm, dd = rng.randint(0, 99), rng.randint(1, 12), rng.randint(1, 28)
        body = f"{yy:02d}{mm:02d}{dd:02d}{rng.randint(1, 6)}{rng.randint(0, 9999):04d}"
        c = _iin_body_check(body)
        if c is not None:
            return body + str(c)


def make_bin(rng: random.Random) -> str:
    while True:
        body = (
            f"{rng.randint(0, 99):02d}{rng.randint(1, 12):02d}"
            f"{rng.choice('456')}{rng.choice('0123')}{rng.randint(0, 99999):05d}"
        )
        c = _iin_body_check(body)
        if c is not None:
            return body + str(c)


def make_card(rng: random.Random) -> str:
    prefix = rng.choice(["4", "51", "52", "53", "5", "2204", "62"])
    body = prefix + "".join(rng.choice(string.digits) for _ in range(15 - len(prefix)))
    # Luhn check digit: double digits at odd distance from the check position
    total = 0
    for i, ch in enumerate(reversed(body)):
        d = int(ch)
        if i % 2 == 0:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return body + str((10 - total % 10) % 10)


def make_iban(rng: random.Random) -> str:
    bban = "".join(rng.choice(string.digits + string.ascii_uppercase) for _ in range(16))
    rearranged = bban + "KZ00"
    num = int("".join(str(int(c, 36)) for c in rearranged))
    check = 98 - num % 97
    return f"KZ{check:02d}{bban}"


def make_email(rng: random.Random) -> str:
    user = rng.choice(["ivan.petrov", "aigerim_s", "test.user", "dev-ops", "m.kassymov", "anna99"])
    dom = rng.choice(["example.com", "mail.example.kz", "corp.example.org", "test.example.ru"])
    return f"{user}{rng.randint(1, 99)}@{dom}"


def make_phone(rng: random.Random) -> str:
    area = rng.choice(["701", "702", "705", "707", "747", "771", "775", "777", "778", "495", "912"])
    a, b, c = f"{rng.randint(0, 999):03d}", f"{rng.randint(0, 99):02d}", f"{rng.randint(0, 99):02d}"
    fmt = rng.randint(0, 6)
    return [
        f"+7 ({area}) {a}-{b}-{c}",
        f"+7{area}{a}{b}{c}",
        f"8 ({area}) {a}-{b}-{c}",
        f"8{area}{a}{b}{c}",
        f"+7 {area} {a} {b} {c}",
        f"8-{area}-{a}-{b}-{c}",
        f"7{area}{a}{b}{c}",
    ][fmt]


def make_passport(rng: random.Random) -> tuple[str, str]:
    """Returns (full text fragment with keyword, number-only span text)."""
    kind = rng.randint(0, 3)
    if kind == 0:
        n = f"N{rng.randint(0, 99999999):08d}"
        return f"passport {n}", n
    if kind == 1:
        n = f"{rng.randint(0, 999999999):09d}"
        return rng.choice(["удостоверение личности", "ID card"]) + f" {n}", n
    if kind == 2:
        n = f"{rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(0, 999999):06d}"
        return f"паспорт серия {n}", n
    n = f"{rng.randint(0, 999999999):09d}"
    return f"жеке куәлік № {n}", n


# ------------------------------------------------------------------ templates

_T = {
    "ru": [
        "Мой ИИН {X}, помоги заполнить декларацию.",
        "Проверь контрагента с ИИН {X} на долги.",
        "ИИН клиента: {X}. Что дальше?",
    ],
    "kk": [
        "Менің ЖСН {X}, салық декларациясына көмектесіңіз.",
        "Клиенттің ЖСН-і {X}, тексеріп беріңізші.",
    ],
    "en": [
        "My KZ tax ID is {X}, can you help with the form?",
        "Look up the person with IIN {X} please.",
    ],
}
_TB = {
    "ru": ["БИН компании {X}, найди реквизиты.", "Счёт выставлен на БИН {X}."],
    "kk": ["Компанияның БСН {X}, деректемелерін тап."],
    "en": ["Invoice for company BIN {X}.", "Our BIN is {X}, draft a letter."],
}
_TP = {
    "ru": [
        "Позвони мне по номеру {X} завтра.",
        "Контактный телефон: {X}.",
        "Напиши на {X} в WhatsApp.",
    ],
    "kk": ["Маған {X} нөміріне қоңырау шалыңыз.", "Байланыс телефоны: {X}."],
    "en": ["Call me at {X} tomorrow.", "My number is {X}, text me."],
}
_TC = {
    "ru": ["Оплатить картой {X}.", "Номер карты {X}, срок 12/29."],
    "kk": ["Картамен төлеу: {X}.", "Карта нөмірі {X}."],
    "en": ["Pay with card {X}.", "My card number is {X}."],
}
_TI = {
    "ru": ["Переведи на счёт {X}.", "IBAN получателя: {X}."],
    "kk": ["Мына шотқа аудар: {X}.", "Алушының IBAN-ы {X}."],
    "en": ["Transfer to account {X}.", "Recipient IBAN: {X}."],
}
_TE = {
    "ru": ["Отправь отчёт на {X}.", "Моя почта {X}, жду ответ."],
    "kk": ["Есепті {X} поштасына жібер.", "Поштам: {X}."],
    "en": ["Send the report to {X}.", "Reach me via {X}."],
}
_TPP = {
    "ru": ["Мои данные: {X}.", "Клиент предъявил {X}."],
    "kk": ["Менің құжатым: {X}."],
    "en": ["The customer provided {X}.", "Details: {X}."],
}

_NEG = {
    "ru": [
        "Расскажи про налоги ИП в Казахстане за 2025 год.",
        "Заказ №{N} отправлен, ожидайте доставку {D}.",
        "Объясни, как работает контрольная сумма ИИН.",
        "Версия библиотеки 3.12.{n}, дата релиза {D}.",
        "Курс доллара {n}.{n} тенге на сегодня.",
        "Сколько будет {N} умножить на {n}?",
    ],
    "kk": [
        "Қазақстандағы жеке кәсіпкерлердің салығы туралы айтып бер.",
        "Тапсырыс №{N} жөнелтілді, күту мерзімі {D}.",
        "ЖСН бақылау санының қалай жұмыс істейтінін түсіндір.",
        "Кітапхана нұсқасы 3.12.{n}, шыққан күні {D}.",
    ],
    "en": [
        "Explain how a Luhn checksum works.",
        "Order #{N} shipped, expected delivery {D}.",
        "Write a regex for email validation, but don't use any real address.",
        "Release 3.12.{n} was tagged on {D}.",
        "What is {N} divided by {n}?",
        "The user at example dot com asked about phone formats.",
    ],
}


def _hard_negatives(rng: random.Random) -> list[str]:
    """Look-alike strings that must NOT be detected."""
    bad_iin = make_iin(rng)
    bad_iin = bad_iin[:11] + str((int(bad_iin[11]) + 1) % 10)
    bad_card = make_card(rng)
    bad_card = bad_card[:15] + str((int(bad_card[15]) + 1) % 10)
    iban = make_iban(rng)
    bad_iban = iban[:2] + f"{(int(iban[2:4]) + 1) % 100:02d}" + iban[4:]
    return [
        f"Номер {bad_iin} не проходит проверку.",
        f"Card {bad_card} is a typo.",
        f"Счёт {bad_iban} введён с ошибкой.",
        f"Артикул {rng.randint(10**11, 10**12 - 1)}7 на складе.",
        f"Серийный номер {rng.randint(10**8, 10**9 - 1)} устройства.",
        f"Order {rng.randint(10**9, 10**10 - 1)} delayed.",
        "Позвоните на 111-22-33 в городской справочной.",
        "Meeting at 10:30, room 4510, floor 2.",
        f"Tracking: {''.join(rng.choice(string.digits) for _ in range(20))}",
        "support@ is not an address; neither is @handle.",
    ]


def _fill(template: str, value: str) -> tuple[str, int, int]:
    i = template.index("{X}")
    text = template.replace("{X}", value)
    return text, i, i + len(value)


def generate(seed: int = 1234, per_type: int = 24) -> list[Sample]:
    rng = random.Random(seed)
    samples: list[Sample] = []

    def single(entity: str, templates: dict[str, list[str]], gen: object) -> None:
        for _ in range(per_type):
            lang = rng.choice(sorted(templates))
            value = gen()  # type: ignore[operator]
            text, s, e = _fill(rng.choice(templates[lang]), value)
            samples.append(Sample(text, ((s, e, entity),), lang))

    single("IIN", _T, lambda: make_iin(rng))
    single("BIN", _TB, lambda: make_bin(rng))
    single("PHONE", _TP, lambda: make_phone(rng))
    single("CARD", _TC, lambda: _spaced(make_card(rng), rng))
    single("IBAN", _TI, lambda: _iban_fmt(make_iban(rng), rng))
    single("EMAIL", _TE, lambda: make_email(rng))

    for _ in range(per_type):
        lang = rng.choice(sorted(_TPP))
        frag, num = make_passport(rng)
        text, s, _e = _fill(rng.choice(_TPP[lang]), frag)
        ns = text.index(num, s)
        samples.append(Sample(text, ((ns, ns + len(num), "PASSPORT"),), lang))

    # multi-entity texts
    for _ in range(40):
        iin, phone, mail = make_iin(rng), make_phone(rng), make_email(rng)
        text = f"Клиент ИИН {iin}, тел. {phone}, почта {mail}. Подготовь письмо."
        spans = []
        for val, ent in ((iin, "IIN"), (phone, "PHONE"), (mail, "EMAIL")):
            st = text.index(val)
            spans.append((st, st + len(val), ent))
        samples.append(Sample(text, tuple(spans), "ru"))
    for _ in range(20):
        card, iban = make_card(rng), make_iban(rng)
        text = f"Pay {card} or transfer to {iban}, then email {make_email(rng)}"
        spans = []
        for val, ent in ((card, "CARD"), (iban, "IBAN")):
            st = text.index(val)
            spans.append((st, st + len(val), ent))
        mail = text.split("email ")[1]
        st = text.index(mail)
        spans.append((st, st + len(mail), "EMAIL"))
        samples.append(Sample(text, tuple(spans), "en"))

    # negatives
    for lang, items in _NEG.items():
        for tpl in items:
            for _ in range(5):
                text = tpl.format(
                    N=rng.randint(100000, 999999),
                    n=rng.randint(1, 99),
                    D=f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.2026",
                )
                samples.append(Sample(text, (), lang))
    for _ in range(4):
        for text in _hard_negatives(rng):
            samples.append(Sample(text, (), "mixed"))
    rng.shuffle(samples)
    return samples


def _spaced(card: str, rng: random.Random) -> str:
    sep = rng.choice(["", " ", "-"])
    return sep.join(card[i : i + 4] for i in range(0, len(card), 4))


def _iban_fmt(iban: str, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return iban
    return " ".join(iban[i : i + 4] for i in range(0, len(iban), 4))
