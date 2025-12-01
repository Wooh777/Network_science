#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EV / H2 충전소 네트워크 시각화 전용 스크립트 (quick_visualize.py)

- EV or H2 한 종류를 선택해서
- 지정한 연도(들)까지 누적된 네트워크를
  반경 R + k_min 하이브리드 규칙으로 만들고
- PNG로 시각화해서 저장한다.

※ 계산 흐름
  1) 선택한 연료타입(EV/H2) 데이터 로드
  2) 전체 좌표에 대해 radius_neighbors, kNN 한 번만 계산
  3) 요청한 각 연도별로 그래프 구성 + 그림 저장

※ 주의
  - 노드 수가 많고(r=30km) 엣지가 많으면 그림 그리는 데 시간이 좀 걸릴 수 있음
  - max_edges 옵션으로 엣지 일부만 샘플링해서 그릴 수 있음
"""

import argparse
import os
import logging
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

EARTH_RADIUS_KM = 6371.0


# =========================
#  데이터 로드
# =========================

def load_ev_data(ev_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(ev_csv_path)

    if "설치년도" not in df.columns:
        raise ValueError("EV CSV에 '설치년도' 컬럼이 없습니다.")
    for col in ["lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"EV CSV에 '{col}' 컬럼이 없습니다.")

    out = df[["설치년도", "lat", "lon"]].copy()
    out = out.rename(columns={"설치년도": "year"})
    out = out.dropna(subset=["year", "lat", "lon"])
    out["year"] = out["year"].astype(int)
    out = out.sort_values("year").reset_index(drop=True)

    logging.info(
        "Loaded EV data: %d rows, year range = [%s, %s]",
        len(out),
        out["year"].min(),
        out["year"].max(),
    )
    return out


def load_h2_data(h2_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(h2_csv_path)

    # year 컬럼 확인
    if "year" not in df.columns:
        logging.warning("H2 CSV에 'year' 컬럼이 없어 모든 행을 동일한 연도로 처리합니다.")
        df["year"] = 9999

    for col in ["lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"H2 CSV에 '{col}' 컬럼이 없습니다.")

    out = df[["year", "lat", "lon"]].copy()
    out = out.dropna(subset=["year", "lat", "lon"])

    try:
        out["year"] = out["year"].astype(int)
    except:
        out["year"] = 9999

    out = out.sort_values("year").reset_index(drop=True)

    logging.info(
        "Loaded H2 data: %d rows, (detected year range: [%s, %s])",
        len(out),
        out["year"].min(),
        out["year"].max(),
    )
    return out



# =========================
#  전역 이웃 계산
# =========================

def precompute_neighbors(
    coords: np.ndarray,
    radius_km: float,
    k_min: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    coords: (N, 2) = [lat, lon]

    반환:
      - radius_dists_list[i]: 노드 i의 반경R 이웃까지의 거리(rad 배열)
      - radius_idx_list[i]:  노드 i의 반경R 이웃 index 배열
      - knn_dists:  (N, k_min+1)  자기 자신 포함 kNN 거리(rad)
      - knn_idx:    (N, k_min+1)  kNN index
    """
    n = coords.shape[0]
    if n == 0:
        return [], [], np.empty((0, 0)), np.empty((0, 0), dtype=int)

    coords_rad = np.radians(coords)

    logging.info("Precomputing radius neighbors (N=%d)...", n)
    nbrs_radius = NearestNeighbors(
        radius=radius_km / EARTH_RADIUS_KM,
        metric="haversine"
    ).fit(coords_rad)
    radius_dists_list, radius_idx_list = nbrs_radius.radius_neighbors(coords_rad)

    logging.info("Precomputing kNN neighbors (k_min=%d)...", k_min)
    k_for_knn = min(k_min + 1, n)
    nbrs_knn = NearestNeighbors(
        n_neighbors=k_for_knn,
        metric="haversine"
    ).fit(coords_rad)
    knn_dists, knn_idx = nbrs_knn.kneighbors(coords_rad)

    logging.info("Neighbor precomputation done.")
    return radius_dists_list, radius_idx_list, knn_dists, knn_idx


# =========================
#  연도별 그래프 구성 + 시각화
# =========================

def build_graph_for_year(
    years_all: np.ndarray,
    coords_all: np.ndarray,
    radius_dists_list: List[np.ndarray],
    radius_idx_list: List[np.ndarray],
    knn_dists: np.ndarray,
    knn_idx: np.ndarray,
    year: int,
    radius_km: float = 30.0,
    k_min: int = 3,
    r_max_km: Optional[float] = None,
) -> Tuple[nx.Graph, np.ndarray]:
    """
    주어진 year까지 누적된 노드들로
    하이브리드 네트워크 그래프를 구성한다.

    반환:
      - G: networkx Graph (노드는 0..(n_active-1))
      - coords: 활성 노드의 좌표 배열 (n_active, 2)
    """
    active_mask = (years_all <= year)
    active_indices = np.where(active_mask)[0]
    n_active = active_indices.size

    if n_active == 0:
        logging.warning("No active nodes for year=%d", year)
        G_empty = nx.Graph()
        return G_empty, np.zeros((0, 2))

    coords = coords_all[active_indices]
    G = nx.Graph()
    G.add_nodes_from(range(n_active))

    # 전역 index → local index 매핑 (빠른 lookup용)
    global_to_local = {g: i for i, g in enumerate(active_indices)}

    # 1) radius 엣지 추가
    for g_i in active_indices:
        local_i = global_to_local[g_i]
        dlist = radius_dists_list[g_i]
        ilist = radius_idx_list[g_i]

        for d_rad, g_j in zip(dlist, ilist):
            if g_j == g_i:
                continue
            if g_j not in global_to_local:
                continue
            local_j = global_to_local[g_j]
            d_km = d_rad * EARTH_RADIUS_KM
            G.add_edge(local_i, local_j, weight=d_km)

    # 2) k_min 보정 (degree < k_min 노드에 대해 kNN 사용)
    for g_i in active_indices:
        local_i = global_to_local[g_i]
        if G.degree[local_i] >= k_min:
            continue

        d_knn_i = knn_dists[g_i][1:]     # 자기 자신 제외
        idx_knn_i = knn_idx[g_i][1:]

        current_deg = G.degree[local_i]
        for d_rad, g_j in zip(d_knn_i, idx_knn_i):
            d_km = d_rad * EARTH_RADIUS_KM
            if r_max_km is not None and d_km > r_max_km:
                continue
            local_j = global_to_local[g_j]
            d_km = d_rad * EARTH_RADIUS_KM
            if r_max_km is not None and d_km > r_max_km:
                continue

            if not G.has_edge(local_i, local_j):
                G.add_edge(local_i, local_j, weight=d_km)
                current_deg += 1
                if current_deg >= k_min:
                    break

    return G, coords


def plot_network(
    coords: np.ndarray,
    G: nx.Graph,
    out_path: str,
    title: str = "",
    dpi: int = 200,
):
    """
    lat/lon 기반 네트워크 시각화.
    모든 엣지를 100% 그대로 그림.
    """
    if coords.shape[0] == 0:
        logging.warning("No nodes to plot for %s", title)
        return

    lats = coords[:, 0]
    lons = coords[:, 1]

    plt.figure(figsize=(8, 10))

    # ⚡ 엣지 100% 그대로 모두 그림
    edges_to_draw = G.edges()

    for u, v in edges_to_draw:
        x = [lons[u], lons[v]]
        y = [lats[u], lats[v]]
        plt.plot(x, y, linewidth=0.2, alpha=0.15, color="blue")

    # 노드
    plt.scatter(lons, lats, s=5, alpha=0.8, color="black")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    logging.info("Saved FULL-EDGE network plot: %s", out_path)



# =========================
#  main
# =========================

def parse_years(years_str: str) -> List[int]:
    """
    "2015,2018,2020" 같은 문자열을 [2015, 2018, 2020]으로 변환
    """
    years = []
    for part in years_str.split(","):
        part = part.strip()
        if not part:
            continue
        years.append(int(part))
    return years


def parse_args():
    parser = argparse.ArgumentParser(
        description="EV/H2 네트워크 시각화 전용 스크립트 (quick_visualize.py)"
    )
    parser.add_argument("--ev-csv", help="EV CSV 경로 (EVdata.csv)")
    parser.add_argument("--h2-csv", help="H2 CSV 경로 (H2data.csv)")
    parser.add_argument("--fuel", choices=["EV", "H2"], required=True,
                        help="시각화할 연료 타입: EV 또는 H2")
    parser.add_argument("--years", required=True,
                        help="시각화할 연도 목록 (예: '2015,2018,2024')")
    parser.add_argument("--out-dir", required=True, help="PNG 저장 디렉토리")
    parser.add_argument("--radius-km", type=float, default=30.0,
                        help="서비스 반경 (km), 기본=30")
    parser.add_argument("--k-min", type=int, default=3,
                        help="하이브리드 최소 degree k_min, 기본=3")
    parser.add_argument("--r-max-km", type=float, default=None,
                        help="kNN 보정 시 최대 허용 거리 (km), 기본=None=제한 없음")
    parser.add_argument("--max-edges", type=int, default=50000,
                        help="시각화 시 그릴 최대 엣지 수 (샘플링), 기본=50000")
    parser.add_argument("--log-level", default="INFO",
                        help="로그 레벨 (DEBUG, INFO, WARNING 등)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    years_to_plot = sorted(set(parse_years(args.years)))

    # 1) 데이터 로드
    if args.fuel == "EV":
        if not args.ev_csv:
            raise ValueError("--fuel EV를 선택하면 --ev-csv를 반드시 지정해야 합니다.")
        df = load_ev_data(args.ev_csv)
    else:  # H2
        if not args.h2_csv:
            raise ValueError("--fuel H2를 선택하면 --h2-csv를 반드시 지정해야 합니다.")
        df = load_h2_data(args.h2_csv)

    coords_all = df[["lat", "lon"]].values
    years_all = df["year"].values

    # 2) 전역 이웃 한 번만 계산
    radius_dists_list, radius_idx_list, knn_dists, knn_idx = precompute_neighbors(
        coords_all,
        radius_km=args.radius_km,
        k_min=args.k_min,
    )

    # 3) 요청된 각 연도에 대해 그래프 구성 + 시각화
    for year in years_to_plot:
        logging.info("[%s] Visualizing network for year <= %d ...", args.fuel, year)
        G, coords = build_graph_for_year(
            years_all=years_all,
            coords_all=coords_all,
            radius_dists_list=radius_dists_list,
            radius_idx_list=radius_idx_list,
            knn_dists=knn_dists,
            knn_idx=knn_idx,
            year=year,
            radius_km=args.radius_km,
            k_min=args.k_min,
            r_max_km=args.r_max_km,
        )
        n_active = coords.shape[0]
        title = f"{args.fuel} network up to {year} (N={n_active})"
        out_path = os.path.join(args.out_dir, f"{args.fuel}_network_{year}.png")

        plot_network(
            coords=coords,
            G=G,
            out_path=out_path,
            title=title,
            dpi=200,
        )


if __name__ == "__main__":
    main()


# python3 visualization.py --fuel H2 --h2-csv H2data.csv --years 2025 --out-dir vis_h2 --radius-km 30 --k-min 3 --r-max-km 50 --log-level INFO
