# %% [markdown]
# # 서울의 "화장실 사막": 낮과 밤의 공중화장실 접근성
# 과제3 — 도시 현상을 지표로 측정하고 공간적으로 해석하기
#
# 실행: `"$HOME/Downloads/2026-2 스마트시티 융합기술세미나/jslee-code/.venv/bin/python" analysis.py`
# (수업 저장소 venv: geopandas, osmnx 사용. `# %%` 단위로 VS Code에서 셀 실행 가능)

# %%
from pathlib import Path

import json

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from hours import is_open, parse_hours

plt.rc("font", family="Apple SD Gothic Neo")
plt.rc("axes", unicode_minus=False, edgecolor="#bbb", linewidth=0.6, titleweight="bold", labelcolor="#555")
plt.rc("text", color="#111")
plt.rc(("xtick", "ytick"), color="#555")
# 그림은 핑크 단색 계열: 연핑크(바탕·비교) → 핑크(강조)
INK, GRAY, LIGHT = "#ee6aa7", "#f08cbf", "#f5d0e2"
PINK = "#e4007f"  # 강조색: 밤·사막
# 연핑크 → 진핑크: 진할수록 상황이 나쁨
from matplotlib.colors import LinearSegmentedColormap
RAMP = LinearSegmentedColormap.from_list("pink", ["#fdf0f6", "#f7bfd9", "#ee6aa7", "#e4007f", "#8c0050"])

COURSE = Path.home() / "Downloads/2026-2 스마트시티 융합기술세미나/jslee-code"
OUT = Path("public")
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

METRIC_CRS = "EPSG:5179"
DAY, NIGHT = 14, 3  # 생활인구 최대(14시)·최소(03~04시) 시각
WALK_M_PER_MIN = 4.0 * 1000 / 60  # 수업과 동일: 시속 4km
BETA = np.log(2) / 10  # 10분 거리 기여도 0.5
MAX_MIN = 30
MAX_M = MAX_MIN * WALK_M_PER_MIN

# %% [markdown]
# ## 1. 데이터
# - 공중화장실: 서울 열린데이터광장 「서울시 공중화장실 위치정보」 OpenAPI(mgisToiletPoi), 4,455개소
# - 생활인구: 수업 7주차 `SPOP_2025_ADM_mid.csv` (2025년 행정동·시간대별 생활인구 중앙값)
# - 행정동 경계: `HangJeongDong_ver20250101.geojson` (생활인구 행정동 코드와 일치)
# - 100m 격자 인구: 수업 6주차 `2024년_인구_다사_100M.csv` (행정동 내부 분포 가중치)
# - 보행 네트워크: OpenStreetMap (OSMnx, walk)

# %%
toilets = pd.read_csv("data/seoul_toilets.csv", encoding="utf-8-sig")
toilets["hours"] = toilets["VALUE_02"].map(parse_hours)
toilets["hours_unknown"] = toilets["hours"].isna()
# 시간 정보를 읽을 수 없는 화장실은 주간 개방·심야 미개방으로 간주
toilets["open_day"] = toilets["hours"].map(lambda h: is_open(h, DAY, unknown=True))
toilets["open_night"] = toilets["hours"].map(lambda h: is_open(h, NIGHT))
toilets["type"] = toilets["VALUE_08"].str.strip(" |").replace("", "미상")
toilets = gpd.GeoDataFrame(
    toilets,
    geometry=gpd.points_from_xy(toilets["COORD_X"], toilets["COORD_Y"]),
    crs="EPSG:4326",
).to_crs(METRIC_CRS)

dong = gpd.read_file(COURSE / "lecture06_Living_Population/data/HangJeongDong_ver20250101.geojson")
dong = dong[dong["sido"] == "11"].to_crs(METRIC_CRS)
dong["ADMI_CD"] = dong["adm_cd2"].str[:8]
dong["gu"] = dong["sggnm"]
dong["dong_nm"] = dong["adm_nm"].str.split().str[-1]
dong = dong[["ADMI_CD", "gu", "dong_nm", "geometry"]].reset_index(drop=True)

spop = pd.read_csv(
    COURSE / "lecture06_Living_Population/data/SPOP_2025_ADM_mid.csv",
    dtype={"ADMI_CD": str, "TT_str": str},
)
spop = spop.pivot_table(index="ADMI_CD", columns="TT_str", values="SPOP_mid")
dong["spop_day"] = dong["ADMI_CD"].map(spop[f"{DAY:02d}"])
dong["spop_night"] = dong["ADMI_CD"].map(spop[f"{NIGHT:02d}"])
assert dong[["spop_day", "spop_night"]].notna().all().all()

# 화장실 → 행정동
toilets = gpd.sjoin(toilets, dong[["ADMI_CD", "geometry"]], how="left", predicate="within")
toilets = toilets.drop(columns="index_right")

print(f"화장실 {len(toilets):,}개, 행정동 {len(dong)}개")
print(f"주간({DAY}시) 개방 {toilets.open_day.sum():,} / 심야({NIGHT}시) 개방 {toilets.open_night.sum():,}")
print(f"개방시간 판독 불가 {toilets.hours_unknown.sum()}개 ({toilets.hours_unknown.mean():.1%})")

# %%
# 100m 격자 거주인구 (행정동 내부에서 사람이 어디에 있는지 가중치)
grid = gpd.read_file(COURSE / "lecture05_Urban_Planning_Index/grid/seoul_100m_grid_2025.geojson")
pop = pd.read_csv(
    COURSE / "lecture05_Urban_Planning_Index/grid/2024년_인구_다사_100M.csv",
    encoding="cp949", header=None, names=["year", "GRID_CD", "stat", "population"],
    dtype={"GRID_CD": str},
)
pop = pop.loc[pop["stat"] == "to_in_001", ["GRID_CD", "population"]]
grid = grid.merge(pop, on="GRID_CD", how="left").fillna({"population": 0})
grid = grid.to_crs(METRIC_CRS)
grid["geometry"] = grid.geometry.centroid
grid = gpd.sjoin(grid, dong[["ADMI_CD", "geometry"]], how="inner", predicate="within")
grid = grid.drop(columns="index_right")
grid = grid[grid["population"] > 0].reset_index(drop=True)
print(f"인구가 있는 100m 격자 {len(grid):,}개")

# %% [markdown]
# ## 2. 지표
#
# **지표 1. 중력형 보행 접근성** (수업 03_accessibility와 같은 설정)
#
# $A_i(t) = \sum_j \exp(-\beta\, t_{ij})$, $t_{ij} \le 30$분, $\beta=\ln 2/10$
#
# - $i$: 100m 격자 중심, $j$: 시각 $t$에 문을 연 화장실, $t_{ij}$: OSM 보행망 최단거리 ÷ 시속 4km
# - 행정동 값은 격자 거주인구로 가중평균: $A_d = \sum P_i A_i / \sum P_i$
# - 해석: 10분 거리 화장실 1개 = 0.5점. 값이 클수록 걸어서 닿는 화장실이 가깝고 많음
#
# **지표 2. 생활인구 1만 명당 개방 화장실 수 (Density)**
#
# $D_d(t) = \dfrac{\text{행정동 } d\text{ 안에서 시각 } t\text{에 개방한 화장실 수}}{\text{생활인구}_d(t)} \times 10{,}000$
#
# **지표 3. 심야 소실률 (직접 설계)**
#
# $L_d = 1 - A_d(\text{03시}) / A_d(\text{14시})$ — 밤이 되면 접근성이 몇 % 사라지는가
#
# **지표 4. 지니계수**: 행정동 간 접근성 불평등(생활인구 가중). 0=완전 평등, 1=완전 불평등

# %%
# 보행망 파일이 있으면 네트워크 거리, 없으면 직선거리(수업 03_accessibility의 1번 방식)
GRAPH = Path("data/seoul_walk_5179.graphml")
DISTANCE = "network" if GRAPH.exists() else "straight"
print("거리 산정:", DISTANCE)

A_day = np.zeros(len(grid))
A_night = np.zeros(len(grid))


def add(rows, meters, o_day, o_night):
    minutes = meters / WALK_M_PER_MIN
    ok = minutes <= MAX_MIN
    w = np.exp(-BETA * minutes[ok])
    if o_day:
        A_day[rows[ok]] += w
    if o_night:
        A_night[rows[ok]] += w


if DISTANCE == "straight":
    from scipy.spatial import cKDTree

    gxy = np.c_[grid.geometry.x, grid.geometry.y]
    tree = cKDTree(gxy)
    for pt, o_day, o_night in zip(toilets.geometry, toilets["open_day"], toilets["open_night"]):
        rows = np.asarray(tree.query_ball_point([pt.x, pt.y], MAX_M), dtype=int)
        if len(rows):
            add(rows, np.hypot(gxy[rows, 0] - pt.x, gxy[rows, 1] - pt.y), o_day, o_night)
else:
    G = ox.convert.to_undirected(ox.load_graphml(GRAPH))
    nodes = ox.graph_to_gdfs(G, edges=False)[["x", "y"]]

    def snap(points):
        """점 → 가장 가까운 보행망 노드와 그 사이 직선거리(m)."""
        ids = ox.distance.nearest_nodes(G, X=points.x.to_numpy(), Y=points.y.to_numpy())
        xy = nodes.loc[ids, ["x", "y"]].to_numpy()
        return np.asarray(ids), np.hypot(points.x.to_numpy() - xy[:, 0], points.y.to_numpy() - xy[:, 1])

    grid_nodes, grid_snap = snap(grid.geometry)
    toilet_nodes, toilet_snap = snap(toilets.geometry)
    node_rows = pd.Series(np.arange(len(grid))).groupby(grid_nodes).apply(np.asarray).to_dict()

    # ponytail: 화장실마다 다익스트라 1회(30분 컷오프). 수천 개 수준이면 충분, 더 커지면 노드 단위로 묶을 것
    for node, offset, o_day, o_night in zip(
        toilet_nodes, toilet_snap, toilets["open_day"], toilets["open_night"]
    ):
        if not (o_day or o_night) or offset > MAX_M:
            continue
        dist = nx.single_source_dijkstra_path_length(G, node, cutoff=MAX_M - offset, weight="length")
        for n, d in dist.items():
            rows = node_rows.get(n)
            if rows is not None:
                add(rows, d + offset + grid_snap[rows], o_day, o_night)

grid["A_day"], grid["A_night"] = A_day, A_night
assert (A_night <= A_day + 1e-9).all(), "심야 개방 화장실은 주간에도 개방으로 분류되어야 함"

# %%
# 보조 지표(지도 표시용): 가장 가까운 개방 화장실까지 도보 시간(분)
from scipy.spatial import cKDTree

gxy = np.c_[grid.geometry.x, grid.geometry.y]
for col, flag in [("walk_day", "open_day"), ("walk_night", "open_night")]:
    open_pts = toilets[toilets[flag]].geometry
    straight, _ = cKDTree(np.c_[open_pts.x, open_pts.y]).query(gxy)
    if DISTANCE == "network":
        # 가상 출발점 → 개방 화장실 노드(가중치=스냅 거리)로 다중 출발 다익스트라 1회
        H = G.copy()
        sel = toilets[flag].to_numpy()
        for n, off in pd.Series(toilet_snap[sel]).groupby(toilet_nodes[sel]).min().items():
            H.add_edge("SRC", n, length=off)
        d = nx.single_source_dijkstra_path_length(H, "SRC", weight="length")
        net = np.array([d.get(n, np.inf) for n in grid_nodes]) + grid_snap
        # 보행망에 닿지 않는 격자(섬 등)는 직선거리로 대체
        print(f"{col}: 보행망 미연결 격자 {np.isinf(net).sum()}개")
        straight = np.where(np.isinf(net), straight, np.maximum(net, straight))
    grid[col] = straight / WALK_M_PER_MIN
assert (grid["walk_night"] >= grid["walk_day"] - 1e-9).all()


def wmean(g, col):
    return np.average(g[col], weights=g["population"])


agg = grid.groupby("ADMI_CD").apply(
    lambda g: pd.Series({c: wmean(g, c) for c in ["A_day", "A_night", "walk_day", "walk_night"]})
)
dong = dong.merge(agg, on="ADMI_CD", how="left")

cnt = toilets.groupby("ADMI_CD")[["open_day", "open_night"]].sum()
dong["n_day"] = dong["ADMI_CD"].map(cnt["open_day"]).fillna(0).astype(int)
dong["n_night"] = dong["ADMI_CD"].map(cnt["open_night"]).fillna(0).astype(int)
dong["D_day"] = dong["n_day"] / dong["spop_day"] * 1e4
dong["D_night"] = dong["n_night"] / dong["spop_night"] * 1e4
dong["night_loss"] = 1 - dong["A_night"] / dong["A_day"]
dong["night_ratio"] = dong["spop_night"] / dong["spop_day"]


def gini(x, w):
    """가중 지니계수(로렌츠 곡선 아래 면적 이용)."""
    order = np.argsort(x)
    x, w = np.asarray(x)[order], np.asarray(w)[order]
    cw = np.cumsum(w) / w.sum()
    cx = np.cumsum(x * w) / (x * w).sum()
    return 1 - np.sum((cx + np.r_[0, cx[:-1]]) * np.diff(np.r_[0, cw]))


assert abs(gini(np.ones(5), np.ones(5))) < 1e-12
assert abs(gini(np.array([0, 0, 0, 1.0]), np.ones(4)) - 0.75) < 1e-12

valid = dong.dropna(subset=["A_day", "A_night"])
G_A_day = gini(valid["A_day"], valid["spop_day"])
G_A_night = gini(valid["A_night"], valid["spop_night"])
G_D_day = gini(dong["D_day"], dong["spop_day"])
G_D_night = gini(dong["D_night"], dong["spop_night"])
print(f"Gini 접근성  주간 {G_A_day:.3f} → 심야 {G_A_night:.3f}")
print(f"Gini 밀도    주간 {G_D_day:.3f} → 심야 {G_D_night:.3f}")
print(dong[["A_day", "A_night", "D_day", "D_night", "night_loss"]].describe().round(2))

# %% [markdown]
# ### 화장실 사막 정의
# 심야 접근성 하위 25% **그리고** 심야 생활인구 상위 50% → "밤의 화장실 사막"
# (사람은 많은데 밤에 걸어서 닿는 화장실이 없는 곳)

# %%
q_low = valid["A_night"].quantile(0.25)
p_med = dong["spop_night"].median()
dong["desert"] = (dong["A_night"] <= q_low) & (dong["spop_night"] >= p_med)
deserts = dong[dong["desert"]].sort_values("A_night")
print(f"심야 화장실 사막 {len(deserts)}개 동")
print(deserts[["gu", "dong_nm", "spop_night", "A_day", "A_night", "night_loss", "n_night"]].round(2).head(20).to_string())

# %% [markdown]
# ## 3. 시각화

# %%
def base(ax, title):
    dong.boundary.plot(ax=ax, color=LIGHT, linewidth=0.3)
    ax.set_title(title, fontsize=14)
    ax.set_axis_off()


# 그림 1. 낮과 밤의 개방 화장실 위치
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
for ax, col, title, color in [
    (axes[0], "open_day", f"{DAY}시 개방 화장실 ({toilets.open_day.sum():,}개)", INK),
    (axes[1], "open_night", f"{NIGHT}시 개방 화장실 ({toilets.open_night.sum():,}개)", PINK),
]:
    base(ax, title)
    toilets[toilets[col]].plot(ax=ax, markersize=1.5, color=color, alpha=0.8)
plt.tight_layout()
plt.savefig(IMG / "01_toilets_day_night.png", dpi=150, bbox_inches="tight")
plt.close()

# 그림 2. 중력형 접근성 주간 vs 심야 (같은 색 범위)
vmax = valid["A_day"].quantile(0.95)
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
for ax, col, title in [
    (axes[0], "A_day", f"보행 접근성 · {DAY}시"),
    (axes[1], "A_night", f"보행 접근성 · 새벽 {NIGHT}시"),
]:
    dong.plot(column=col, ax=ax, cmap=RAMP.reversed(), vmin=0, vmax=vmax, edgecolor="white",
              linewidth=0.2, missing_kwds={"color": "lightgrey"})
    base(ax, title)
sm = plt.cm.ScalarMappable(cmap=RAMP.reversed(), norm=plt.Normalize(0, vmax))
fig.colorbar(sm, ax=axes, shrink=0.6, label="중력형 접근성 점수 (10분 거리 1곳 = 0.5)")
plt.savefig(IMG / "02_access_day_night.png", dpi=150, bbox_inches="tight")
plt.close()

# 그림 3. 심야 소실률 + 화장실 사막
fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
dong.plot(column="night_loss", ax=axes[0], cmap=RAMP, vmin=0, vmax=1, legend=True,
          legend_kwds={"shrink": 0.6, "label": "1 - 심야/주간 접근성"},
          edgecolor="white", linewidth=0.2, missing_kwds={"color": "lightgrey"})
base(axes[0], "행정동별 심야 소실률")
dong.plot(ax=axes[1], color="#fdf0f6", edgecolor="white", linewidth=0.2)
dong[dong.desert].plot(ax=axes[1], color=PINK, edgecolor="white", linewidth=0.2)
base(axes[1], f"밤의 화장실 사막 ({dong.desert.sum()}개 동)\n심야 접근성 하위 25% × 심야 생활인구 상위 50%")
plt.savefig(IMG / "03_night_loss_desert.png", dpi=150, bbox_inches="tight")
plt.close()

# 그림 4. 로렌츠 곡선 (비공간): 접근성 vs 1인당 공급
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
for ax, (title, specs) in zip(axes, [
    ("지표 1 · 중력형 보행 접근성", [
        ("A_day", "spop_day", f"{DAY}시 (Gini {G_A_day:.2f})", GRAY),
        ("A_night", "spop_night", f"{NIGHT}시 (Gini {G_A_night:.2f})", PINK)]),
    ("지표 2 · 생활인구 1만 명당 개방 화장실", [
        ("D_day", "spop_day", f"{DAY}시 (Gini {G_D_day:.2f})", GRAY),
        ("D_night", "spop_night", f"{NIGHT}시 (Gini {G_D_night:.2f})", PINK)]),
]):
    for col, wcol, label, color in specs:
        v = dong.dropna(subset=[col]).sort_values(col)
        cw = np.r_[0, np.cumsum(v[wcol]) / v[wcol].sum()]
        cx = np.r_[0, np.cumsum(v[col] * v[wcol]) / (v[col] * v[wcol]).sum()]
        ax.plot(cw, cx, label=label, color=color, linewidth=1.6)
    ax.plot([0, 1], [0, 1], ":", color=GRAY, linewidth=1, label="완전 평등")
    ax.set_xlabel("생활인구 누적 비율 (값이 낮은 동부터)")
    ax.set_ylabel("지표값 누적 비율")
    ax.set_title(title)
    ax.legend()
plt.tight_layout()
plt.savefig(IMG / "04_lorenz.png", dpi=150, bbox_inches="tight")
plt.close()

# 그림 5. 시설 유형별 24시간 개방 비율 (비공간)
by_type = toilets.groupby("type").agg(n=("open_night", "size"), night=("open_night", "mean"))
by_type = by_type[by_type["n"] >= 30].sort_values("night")
fig, ax = plt.subplots(figsize=(8, 5))
# 밤에 대부분 닫는 유형(25% 미만)을 강조
hi = [PINK if r < 0.25 else GRAY for r in by_type["night"]]
ax.barh(by_type.index, by_type["night"] * 100, color=hi, height=0.6)
for i, (n, r) in enumerate(zip(by_type["n"], by_type["night"])):
    ax.text(r * 100 + 1, i, f"{r:.0%} (n={n})", va="center", fontsize=9, color=hi[i])
for lbl, c in zip(ax.get_yticklabels(), hi):
    lbl.set_color(c)
ax.set_xlim(0, 110)
ax.set_xlabel(f"새벽 {NIGHT}시 개방 비율 (%)")
ax.set_title("시설 유형별 새벽 3시 개방률")
plt.savefig(IMG / "05_type_night.png", dpi=150, bbox_inches="tight")
plt.close()

# 그림 6. 심야 생활인구 vs 심야 접근성 (비공간)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(dong["spop_night"], dong["A_night"], s=12, facecolor="white", edgecolor=GRAY, linewidth=0.6)
ax.scatter(deserts["spop_night"], deserts["A_night"], s=16, color=PINK, label="화장실 사막")
ax.axhline(q_low, color=GRAY, linestyle=":", linewidth=1)
ax.axvline(p_med, color=GRAY, linestyle=":", linewidth=1)
label_rows = pd.concat([
    deserts.nlargest(3, "spop_night"), deserts.nsmallest(3, "A_night")
]).drop_duplicates("ADMI_CD")
for k, (_, r) in enumerate(label_rows.iterrows()):
    ax.annotate(f"{r.gu} {r.dong_nm}", (r.spop_night, r.A_night), fontsize=8,
                xytext=(-60, 14 + 12 * (k % 3)), textcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": "#999", "linewidth": 0.6})
ax.set_xlabel(f"새벽 {NIGHT}시 생활인구")
ax.set_ylabel(f"새벽 {NIGHT}시 보행 접근성")
ax.set_title("행정동별 새벽 3시 생활인구와 보행 접근성")
ax.legend()
plt.savefig(IMG / "06_scatter_desert.png", dpi=150, bbox_inches="tight")
plt.close()

# %%
# 웹 지도·표용 결과 저장
out = dong.to_crs("EPSG:4326").copy()
num = ["spop_day", "spop_night", "A_day", "A_night", "D_day", "D_night", "night_loss", "night_ratio", "walk_day", "walk_night"]
out[num] = out[num].round(3)
out["geometry"] = out.geometry.simplify(0.0003)
out.drop(columns="geometry").to_csv("data/dong_indicators.csv", index=False, encoding="utf-8-sig")
pts = toilets.to_crs("EPSG:4326").loc[toilets["open_night"], ["CONTS_NAME", "type", "VALUE_02", "geometry"]]
pts = pts.assign(VALUE_02=pts["VALUE_02"].str.strip(" |"))
summary = {
    "toilets": len(toilets), "open_day": int(toilets.open_day.sum()),
    "open_night": int(toilets.open_night.sum()), "hours_unknown": int(toilets.hours_unknown.sum()),
    "gini_A_day": round(G_A_day, 3), "gini_A_night": round(G_A_night, 3),
    "gini_D_day": round(G_D_day, 3), "gini_D_night": round(G_D_night, 3),
    "deserts": int(dong.desert.sum()), "distance": DISTANCE,
}
# fetch() 대신 <script>로 읽어 file://(사파리 등)에서도 지도가 열리도록 함
(OUT / "data.js").write_text(
    f"window.DONG = {out.to_json(drop_id=True)};\n"
    f"window.NIGHT_TOILETS = {pts.to_json(drop_id=True)};\n"
    f"window.SUMMARY = {json.dumps(summary, ensure_ascii=False)};\n",
    encoding="utf-8",
)
print("저장 완료:", sorted(p.name for p in IMG.iterdir()))
