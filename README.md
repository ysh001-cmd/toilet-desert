# 서울의 화장실 사막

스마트시티 융합기술세미나 과제3: 서울 공중화장실의 낮(14시)과 밤(03시) 보행 접근성을 행정동 단위로 측정합니다.

## 지표
1. 중력형 보행 접근성: `A_i = Σ exp(-β t_ij)`, β = ln2/10, 최대 30분, 4km/h, 격자 인구 가중평균
2. 생활인구 1만 명당 개방 화장실 수
3. 심야 소실률: `1 - A(03시) / A(14시)`
4. 지니계수 (생활인구 가중)

## 실행
수업 저장소(`~/Downloads/2026-2 스마트시티 융합기술세미나/jslee-code`)의 venv와 데이터를 사용합니다.

```bash
python3 hours.py                                   # 개방시간 파서 자체 점검
"$HOME/Downloads/2026-2 스마트시티 융합기술세미나/jslee-code/.venv/bin/python" analysis.py  # 지표 계산 + public/ 그림·데이터 생성
```

`data/seoul_walk_5179.graphml`(OSM 보행망)이 있으면 네트워크 거리, 없으면 직선거리를 씁니다.

## 구성
- `analysis.py`: 분석 전체 (`# %%` 셀 단위 실행 가능)
- `hours.py`: 개방시간 문자열 → 시각별 개방 여부
- `data/seoul_toilets.csv`: 서울 열린데이터광장 `mgisToiletPoi` (공공누리 1유형)
- `public/`: Cloudflare Pages로 배포하는 결과 페이지

## 출처
서울특별시 공중화장실 위치정보 · 서울 생활인구(2025) · 통계청 SGIS 100m 격자 인구(2024) · vuski/admdongkor · © OpenStreetMap contributors
