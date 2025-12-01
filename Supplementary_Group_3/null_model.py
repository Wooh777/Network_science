#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EV/H2 Null Model 분석 스크립트

- 실제 네트워크와 동일한 노드 수(N)를
  같은 bounding box 안에 무작위로 배치한 뒤
  radius + k_min 규칙으로 네트워크를 만들고
  coverage / 고립 / degree / GCC 비율을 계산한다.

- 이를 여러 번 반복(n_trials)하여
  평균/표준편차를 CSV로 저장한다.

입력:
  - EV CSV: 설치년도, lat, lon 포함 (EVdata.csv)
  - H2 CSV: year, lat, lon 포함 (H2data.csv)

출력:
  - ev_null_stats.csv
  - h2_null_stats.csv
"""

import argparse
import os
import logging
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors

EARTH_RADIUS_KM = 6371.0


# =========================
#  공통: 데이터 로드
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

    for col in ["year", "lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"H2 CSV에 '{col}' 컬럼이 없습니다.")

    out = df[["year", "lat", "lon"]].copy()
    out = out.dropna(subset=["year", "lat", "lon"])
    out["year"] = out["year"].astype(int)
    out = out.sort_values("year").reset_index(drop=True)

    logging.info(
        "Loaded H2 data: %d rows, year range = [%s, %s]",
        len(out),
        out["year"].min(),
        out["year"].max(),
    )
    return out


# =========================
#  네트워크 지표 계산 (coords → metrics)
# =========================

def compute_metrics_from_coords(
    coords: np.ndarray,
    radius_km: float = 30.0,
    k_min: int = 3,
    r_max_km: Optional[float] = None,
) -> Dict[str, float]:
    """
    coords: (N, 2) = [lat, lon]

    - radius_km 안의 이웃들로 coverage / 고립 계산
    - 그래프 엣지는 radius + k_min 보정으로 구성
    - avg_path_length / clustering 은 계산하지 않음
    """
    n = coords.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "coverage_ratio": 0.0,
            "n_isolated_R": 0,
            "avg_degree": 0.0,
            "gcc_ratio": 0.0,
        }

    coords_rad = np.radians(coords)

    # 1) radius 이웃
    nbrs_radius = NearestNeighbors(
        radius=radius_km / EARTH_RADIUS_KM,
        metric="haversine"
    ).fit(coords_rad)
    dist_list, idx_list = nbrs_radius.radius_neighbors(coords_rad)

    within_R_counts = np.array([len(idx) - 1 for idx in idx_list])  # 자기 자신 제외
    coverage_ratio = float(np.sum(within_R_counts >= 3)) / float(n)
    n_isolated_R = int(np.sum(within_R_counts < 3))

    for i in range(n):
        dists_i = dist_list[i]
        idx_i = idx_list[i]
        for d_rad, j in zip(dists_i, idx_i):
            if i == j:
                continue
            d_km = d_rad * EARTH_RADIUS_KM
            G.add_edge(i, j, weight=d_km)

    # 2) k_min 보정 (kNN)
    k_for_knn = min(k_min + 1, n)
    nbrs_knn = NearestNeighbors(
        n_neighbors=k_for_knn,
        metric="haversine"
    ).fit(coords_rad)
    d_knn, idx_knn = nbrs_knn.kneighbors(coords_rad)

    for i in range(n):
        current_deg = G.degree[i]
        if current_deg >= k_min:
            continue
        for d_rad, j in zip(d_knn[i][1:], idx_knn[i][1:]):  # 자기 자신 제외
            d_km = d_rad * EARTH_RADIUS_KM
            if r_max_km is not None and d_km > r_max_km:
                continue
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=d_km)
                current_deg += 1
                if current_deg >= k_min:
                    break

    n_nodes = n
    n_edges = G.number_of_edges()

    degrees = np.array([deg for _, deg in G.degree()])
    avg_degree = float(degrees.mean()) if n_nodes > 0 else 0.0

    components = list(nx.connected_components(G))
    if len(components) == 0:
        gcc_ratio = 0.0
    else:
        largest_comp = max(components, key=len)
        gcc_size = len(largest_comp)
        gcc_ratio = float(gcc_size) / float(n_nodes)

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "coverage_ratio": coverage_ratio,
        "n_isolated_R": n_isolated_R,
        "avg_degree": avg_degree,
        "gcc_ratio": gcc_ratio,
    }


# =========================
#  Null model (랜덤 배치)
# =========================

def run_null_model_for_year(
    df: pd.DataFrame,
    year: int,
    n_trials: int = 100,
    radius_km: float = 30.0,
    k_min: int = 3,
    r_max_km: Optional[float] = None,
) -> Dict[str, float]:
    """
    해당 연도까지(누적) 설치된 노드 개수 N을 기준으로,
    같은 bounding box 안에 N개를 랜덤 배치한 Null 모델을 n_trials번 수행하고,
    지표 평균/표준편차를 반환.
    """
    df_year = df[df["year"] <= year].copy()
    coords_real = df_year[["lat", "lon"]].values
    n = coords_real.shape[0]

    if n == 0:
        logging.warning("  [year=%d] no nodes for null model", year)
        return {
            "year": year,
            "n_nodes": 0,
            "n_trials": n_trials,
            "coverage_mean": 0.0,
            "coverage_std": 0.0,
            "n_isolated_mean": 0.0,
            "n_isolated_std": 0.0,
            "avg_degree_mean": 0.0,
            "avg_degree_std": 0.0,
            "gcc_ratio_mean": 0.0,
            "gcc_ratio_std": 0.0,
        }

    # bounding box (실제 네트워크와 동일 영역)
    lat_min, lat_max = coords_real[:, 0].min(), coords_real[:, 0].max()
    lon_min, lon_max = coords_real[:, 1].min(), coords_real[:, 1].max()

    logging.info("  [year=%d] Null model: N=%d, bbox=[%.4f, %.4f]x[%.4f, %.4f]",
                 year, n, lat_min, lat_max, lon_min, lon_max)

    cov_list: List[float] = []
    iso_list: List[float] = []
    deg_list: List[float] = []
    gcc_list: List[float] = []

    for t in range(n_trials):
        # 랜덤 좌표 생성 (lat, lon 각각 uniform)
        rand_lats = np.random.uniform(lat_min, lat_max, size=n)
        rand_lons = np.random.uniform(lon_min, lon_max, size=n)
        coords_rand = np.column_stack([rand_lats, rand_lons])

        metrics = compute_metrics_from_coords(
            coords_rand,
            radius_km=radius_km,
            k_min=k_min,
            r_max_km=r_max_km,
        )

        cov_list.append(metrics["coverage_ratio"])
        iso_list.append(metrics["n_isolated_R"])
        deg_list.append(metrics["avg_degree"])
        gcc_list.append(metrics["gcc_ratio"])

        if (t + 1) % max(1, n_trials // 5) == 0:
            logging.info("    trial %d/%d done (coverage=%.3f, gcc=%.3f)",
                         t + 1, n_trials, cov_list[-1], gcc_list[-1])

    cov_arr = np.array(cov_list)
    iso_arr = np.array(iso_list)
    deg_arr = np.array(deg_list)
    gcc_arr = np.array(gcc_list)

    result = {
        "year": year,
        "n_nodes": n,
        "n_trials": n_trials,
        "coverage_mean": float(cov_arr.mean()),
        "coverage_std": float(cov_arr.std(ddof=1)),
        "n_isolated_mean": float(iso_arr.mean()),
        "n_isolated_std": float(iso_arr.std(ddof=1)),
        "avg_degree_mean": float(deg_arr.mean()),
        "avg_degree_std": float(deg_arr.std(ddof=1)),
        "gcc_ratio_mean": float(gcc_arr.mean()),
        "gcc_ratio_std": float(gcc_arr.std(ddof=1)),
    }
    return result


def run_null_model_dataset(
    df: pd.DataFrame,
    fuel_type: str,
    out_dir: str,
    n_trials: int = 100,
    radius_km: float = 30.0,
    k_min: int = 3,
    r_max_km: Optional[float] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:

    years_unique = sorted(df["year"].unique())
    if start_year is not None:
        years_unique = [y for y in years_unique if y >= start_year]
    if end_year is not None:
        years_unique = [y for y in years_unique if y <= end_year]

    if not years_unique:
        logging.warning("[%s] No years to process in null model.", fuel_type)
        return pd.DataFrame()

    records = []
    total_years = len(years_unique)

    for idx, year in enumerate(years_unique):
        logging.info("[%s] Null model (%d/%d) year=%d ...",
                     fuel_type, idx + 1, total_years, year)

        res = run_null_model_for_year(
            df=df,
            year=year,
            n_trials=n_trials,
            radius_km=radius_km,
            k_min=k_min,
            r_max_km=r_max_km,
        )
        res["fuel_type"] = fuel_type
        records.append(res)

    stats_df = pd.DataFrame(records)
    stats_df = stats_df.sort_values("year").reset_index(drop=True)

    # CSV 저장
    fname = f"{fuel_type.lower()}_null_stats.csv"
    out_path = os.path.join(out_dir, fname)
    stats_df.to_csv(out_path, index=False)
    logging.info("[%s] Saved null model CSV: %s", fuel_type, out_path)

    return stats_df


# =========================
#  main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EV/H2 Null Model (random placement) 분석"
    )
    parser.add_argument("--ev-csv", required=True, help="EV CSV 경로 (EVdata.csv)")
    parser.add_argument("--h2-csv", required=True, help="H2 CSV 경로 (H2data.csv)")
    parser.add_argument("--out-dir", required=True, help="출력 디렉토리")
    parser.add_argument("--radius-km", type=float, default=30.0,
                        help="서비스 반경 (km), 기본=30")
    parser.add_argument("--k-min", type=int, default=3,
                        help="최소 degree k_min, 기본=3")
    parser.add_argument("--r-max-km", type=float, default=None,
                        help="kNN 보정 시 최대 허용 거리 (km), 기본=None=제한 없음")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Null model 반복 횟수 (기본=50; 속도/안정성 trade-off)")
    parser.add_argument("--start-year", type=int, default=None,
                        help="분석 시작 연도 (옵션)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="분석 종료 연도 (옵션)")
    parser.add_argument("--log-level", default="INFO",
                        help="로그 레벨 (DEBUG, INFO, WARNING, ...)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    logging.info("Loading EV data...")
    ev_df = load_ev_data(args.ev_csv)
    logging.info("Loading H2 data...")
    h2_df = load_h2_data(args.h2_csv)

    logging.info("Running EV null model...")
    run_null_model_dataset(
        ev_df,
        fuel_type="EV",
        out_dir=args.out_dir,
        n_trials=args.n_trials,
        radius_km=args.radius_km,
        k_min=args.k_min,
        r_max_km=args.r_max_km,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    logging.info("Running H2 null model...")
    run_null_model_dataset(
        h2_df,
        fuel_type="H2",
        out_dir=args.out_dir,
        n_trials=args.n_trials,
        radius_km=args.radius_km,
        k_min=args.k_min,
        r_max_km=args.r_max_km,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()

# python3 null_model.py --ev-csv EVdata.csv --h2-csv H2data.csv --out-dir output_null --radius-km 30 --k-min 3 --n-trials 50 --start-year 2017 --end-year 2017 --log-level INFO
# python3 null_model.py --ev-csv EVdata.csv --h2-csv H2data.csv --out-dir output_null --radius-km 30 --k-min 3 --n-trials 50 --start-year 2025 --end-year 2025 --log-level INFO