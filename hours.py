"""공중화장실 개방시간 문자열(VALUE_02) → 특정 시각 개방 여부.

규칙
- '상시(24시간)'이 있으면 항상 개방
- 'HH:MM~HH:MM' 형태의 첫 번째 시간 범위를 평일 개방시간으로 사용
  ('익일', 종료 < 시작, 24:00 초과는 자정을 넘기는 것으로 처리)
- 시간 범위를 읽을 수 없으면(영업시작~종료, 운영시간내, 15시간, 공란 등) None
  → 분석에서는 '주간 개방, 심야 미개방'으로 간주
"""
import re

RANGE = re.compile(
    r"(\d{1,2})(?::(\d{2})|시)?\s*~\s*(?:익일)?\s*(\d{1,2})(?::(\d{2})|시)?"
)


def parse_hours(s):
    """(start_min, end_min) 또는 항상 개방이면 (0, 1440), 모르면 None."""
    s = (s or "").replace("::", ":")
    if "상시" in s or "24시간" in s:
        return (0, 1440)
    m = RANGE.search(s)
    if not m:
        return None
    h1, m1, h2, m2 = m.groups()
    start = int(h1) * 60 + int(m1 or 0)
    end = int(h2) * 60 + int(m2 or 0)
    if start >= 1440:
        return None
    if end <= start or "익일" in s:
        end += 1440  # 자정을 넘김
    return (start, end)


def is_open(hours, hour, unknown=False):
    """hours: parse_hours 결과, hour: 0~23시 정각. 모르는 경우 unknown 반환."""
    if hours is None:
        return unknown
    start, end = hours
    t = hour * 60
    return start <= t < end or start <= t + 1440 < end


if __name__ == "__main__":
    assert parse_hours("상시(24시간)|") == (0, 1440)
    assert parse_hours("정시(09:00~18:00,평일)|") == (540, 1080)
    assert parse_hours("기타|05:00~익일01:00|") == (300, 1500)
    assert parse_hours("기타|10:00~05:00|") == (600, 1740)
    assert parse_hours("기타|06:00~25:00|") == (360, 1500)
    assert parse_hours("기타|07~19시|법정공휴일 미개방|") == (420, 1140)
    assert parse_hours("기타|06::00~23:00|") == (360, 1380)
    assert parse_hours("정시(영업시작~종료)|") is None
    assert parse_hours("15시간") is None
    assert parse_hours(" ") is None

    assert is_open((0, 1440), 3)
    assert is_open((540, 1080), 14) and not is_open((540, 1080), 3)
    assert is_open((300, 1500), 0) and not is_open((300, 1500), 3)  # 05~익일01
    assert is_open((600, 1740), 3)  # 10:00~05:00
    assert is_open(None, 14, unknown=True) and not is_open(None, 3)
    print("hours.py ok")
