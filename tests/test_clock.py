from datetime import date, datetime, timedelta, timezone

import pytest

from orma import clock

IST = timezone(timedelta(hours=5, minutes=30))
THU = datetime(2026, 10, 15, 14, 0, tzinfo=IST).timestamp()  # Thursday afternoon in India


def day_of(stamp, tz=IST):
    return clock.day(stamp, tz)


def test_zone_names():
    assert clock.zone("") is None
    assert clock.zone("utc") is timezone.utc


def test_day_follows_the_users_zone_not_utc():
    just_after_midnight_ist = datetime(2026, 10, 15, 0, 30, tzinfo=IST).timestamp()  # still the 14th in UTC
    assert clock.day(just_after_midnight_ist, IST) == date(2026, 10, 15)
    assert clock.day(just_after_midnight_ist, timezone.utc) == date(2026, 10, 14)


def test_system_zone_works_even_near_1970():
    assert clock.iso_day(1.0, None) in ("1969-12-31", "1970-01-01")
    assert clock.parse_time("1970-01-01", None, None) is not None


def test_naive_iso_is_read_in_the_users_zone():
    assert clock.parse_time("2026-10-10", None, IST) == datetime(2026, 10, 10, tzinfo=IST).timestamp()
    assert clock.parse_time("2026-10-10T08:00Z", None, IST) == datetime(2026, 10, 10, 8, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize("phrase, expected", [
    ("last Saturday", date(2026, 10, 10)),
    ("Saturday", date(2026, 10, 10)),
    ("yesterday", date(2026, 10, 14)),
    ("3 days ago", date(2026, 10, 12)),
    ("last month", date(2026, 9, 15)),
    ("in March 2024", date(2024, 3, 1)),
    ("since 2019", date(2019, 1, 1)),
    ("2026-10-09", date(2026, 10, 9)),
])
def test_date_words_resolve_against_when_they_were_said(phrase, expected):
    pytest.importorskip("dateparser")
    assert day_of(clock.resolve(phrase, THU, IST)) == expected


def test_resolved_dates_start_at_local_midnight():
    pytest.importorskip("dateparser")
    assert clock.resolve("yesterday", THU, IST) == datetime(2026, 10, 14, tzinfo=IST).timestamp()


def test_date_order_reads_numeric_dates_the_users_way():
    pytest.importorskip("dateparser")
    assert day_of(clock.resolve("08/10/2026", THU, IST, "DMY")) == date(2026, 10, 8)
    assert day_of(clock.resolve("08/10/2026", THU, IST, "MDY")) == date(2026, 8, 10)


def test_unparseable_phrases_are_none():
    pytest.importorskip("dateparser")
    assert clock.resolve("a while back maybe", THU, IST) is None
    assert clock.resolve("", THU, IST) is None
