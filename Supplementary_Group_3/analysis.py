#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EV/H2 충전소 네트워크 지표 분석 (고속 버전)

변경점:
  - EV, H2 각각에 대해 haversine radius / kNN 이웃을 "한 번만" 계산
  - 각 연도에서는 해당 연도까지의 노드만 골라서 서브그래프 구성
  - 시각화(네트워크 그림) 완전 제거 → CSV 지표만 출력

EV CSV 컬럼:
  설치년도,시도,군구,주소,충전소명,시설구분(대),시설구분(소),
  기종(대),기종(소),운영기관(대),운영기관(소),급속충전량,
  충전기타입,이용자제한,위도,경도,lat,lon

H2 CSV 컬럼:
  station_id,year,name,주소,구분,공급방식,lat,lon
"""

import argparse
import os
import logging
from typing import Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp


EARTH_RADIUS_KM = 6371.0


# =========================
#  데이터 로드 & 정규화
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

    # 연도 정렬 (중요! → 연도 누적 시 prefix가 되게)
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
#  전역 이웃 한 번만 계산
# =========================

def precompute_neighbors(
    coords: np.ndarray,
    radius_km: float,
    k_min: int
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

    # 1) 반경 이웃
    logging.info("  - Precomputing radius neighbors (N=%d)...", n)
    nbrs_radius = NearestNeighbors(
        radius=radius_km / EARTH_RADIUS_KM,
        metric="haversine"
    ).fit(coords_rad)
    radius_dists_list, radius_idx_list = nbrs_radius.radius_neighbors(coords_rad)

    # 2) kNN 이웃
    logging.info("  - Precomputing kNN neighbors (k_min=%d)...", k_min)
    k_for_knn = min(k_min + 1, n)  # 자기 자신 포함
    nbrs_knn = NearestNeighbors(
        n_neighbors=k_for_knn,
        metric="haversine"
    ).fit(coords_rad)
    knn_dists, knn_idx = nbrs_knn.kneighbors(coords_rad)

    logging.info("  - Neighbor precomputation done.")
    return radius_dists_list, radius_idx_list, knn_dists, knn_idx


# =========================
#  한 연도에 대한 그래프 + 지표 계산
# =========================

def build_graph_and_metrics_for_year(
    year: int,
    years_all: np.ndarray,
    coords_all: np.ndarray,
    radius_dists_list: List[np.ndarray],
    radius_idx_list: List[np.ndarray],
    knn_dists: np.ndarray,
    knn_idx: np.ndarray,
    radius_km: float = 30.0,
    k_min: int = 3,
    r_max_km: Optional[float] = None,
) -> Dict[str, float]:
    """
    이미 전체 데이터에 대해 이웃이 계산되어 있다는 가정 하에,
    특정 year까지의 노드만 사용하여 그래프를 만들고 지표를 계산.
    """

    # 활성 노드: year <= 현재 year 인 인덱스
    active_mask = (years_all <= year)
    active_indices = np.where(active_mask)[0]
    n_active = active_indices.size

    # 활성 노드가 없으면 바로 리턴
    if n_active == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "coverage_ratio": 0.0,
            "n_isolated_R": 0,
            "avg_degree": 0.0,
            "gcc_ratio": 0.0,
            "avg_path_length_gcc": float("nan"),
            "avg_clustering": float("nan"),
        }

    G = nx.Graph()
    G.add_nodes_from(active_indices.tolist())

    # ----- 1) 반경 R 기반 엣지 + coverage / isolated 계산 -----
    coverage_count = 0
    n_isolated_R = 0

    for i in active_indices:
        dists_i = radius_dists_list[i]
        idx_i = radius_idx_list[i]

        # 활성 노드 중 반경 R 안에 있는 이웃 개수 (자기 자신 제외)
        neighbor_count = 0
        for d_rad, j in zip(dists_i, idx_i):
            if i == j:
                continue
            if not active_mask[j]:
                continue
            neighbor_count += 1
            d_km = d_rad * EARTH_RADIUS_KM
            # 엣지 추가 (양방향이지만 Graph가 중복 제거해줌)
            G.add_edge(i, j, weight=d_km)

        if neighbor_count >= 3:
            coverage_count += 1
        else:
            n_isolated_R += 1

    coverage_ratio = float(coverage_count) / float(n_active)

    # ----- 2) k_min 보정 (degree < k_min 인 노드) -----
    for i in active_indices:
        current_deg = G.degree[i]
        if current_deg >= k_min:
            continue

        # kNN 이웃(자기 자신 제외)
        for d_rad, j in zip(knn_dists[i][1:], knn_idx[i][1:]):
            if not active_mask[j]:
                continue
            d_km = d_rad * EARTH_RADIUS_KM
            if r_max_km is not None and d_km > r_max_km:
                continue
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=d_km)
                current_deg += 1
                if current_deg >= k_min:
                    break

    # =========================
    #  네트워크 지표 계산
    # =========================
    n_nodes = n_active
    n_edges = G.number_of_edges()

    degrees = np.array([deg for _, deg in G.degree()])
    avg_degree = float(degrees.mean()) if n_nodes > 0 else 0.0

    components = list(nx.connected_components(G))
    if len(components) == 0:
        gcc_ratio = 0.0
        avg_path_length_gcc = float("nan")
    else:
        largest_comp = max(components, key=len)
        gcc_size = len(largest_comp)
        gcc_ratio = float(gcc_size) / float(n_nodes)

        # 평균 최단거리 계산은 너무 비싸서 일단 생략 (원하면 샘플링 버전으로 다시 넣자)
        avg_path_length_gcc = float("nan")

    avg_clustering = float("nan")

    metrics = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "coverage_ratio": coverage_ratio,
        "n_isolated_R": n_isolated_R,
        "avg_degree": avg_degree,
        "gcc_ratio": gcc_ratio,
        "avg_path_length_gcc": avg_path_length_gcc,
        "avg_clustering": avg_clustering,
    }
    return metrics

# ===== 전역 변수 (워커에서 재사용) =====
_GLOBAL_years_all = None
_GLOBAL_coords_all = None
_GLOBAL_radius_dists_list = None
_GLOBAL_radius_idx_list = None
_GLOBAL_knn_dists = None
_GLOBAL_knn_idx = None
_GLOBAL_radius_km = None
_GLOBAL_k_min = None
_GLOBAL_r_max_km = None


def _init_worker_year_metrics(
    years_all,
    coords_all,
    radius_dists_list,
    radius_idx_list,
    knn_dists,
    knn_idx,
    radius_km,
    k_min,
    r_max_km,
):
    """
    각 프로세스 시작 시 한 번 호출되어,
    큰 배열들을 전역변수로 등록해둔다.
    """
    global _GLOBAL_years_all, _GLOBAL_coords_all
    global _GLOBAL_radius_dists_list, _GLOBAL_radius_idx_list
    global _GLOBAL_knn_dists, _GLOBAL_knn_idx
    global _GLOBAL_radius_km, _GLOBAL_k_min, _GLOBAL_r_max_km

    _GLOBAL_years_all = years_all
    _GLOBAL_coords_all = coords_all
    _GLOBAL_radius_dists_list = radius_dists_list
    _GLOBAL_radius_idx_list = radius_idx_list
    _GLOBAL_knn_dists = knn_dists
    _GLOBAL_knn_idx = knn_idx
    _GLOBAL_radius_km = radius_km
    _GLOBAL_k_min = k_min
    _GLOBAL_r_max_km = r_max_km


def _worker_compute_year_metrics(year: int) -> Dict[str, float]:
    logging.info("[worker] start year=%d", year)

    metrics = build_graph_and_metrics_for_year(
        year=year,
        years_all=_GLOBAL_years_all,
        coords_all=_GLOBAL_coords_all,
        radius_dists_list=_GLOBAL_radius_dists_list,
        radius_idx_list=_GLOBAL_radius_idx_list,
        knn_dists=_GLOBAL_knn_dists,
        knn_idx=_GLOBAL_knn_idx,
        radius_km=_GLOBAL_radius_km,
        k_min=_GLOBAL_k_min,
        r_max_km=_GLOBAL_r_max_km,
    )

    logging.info("[worker] done year=%d (nodes=%d, edges=%d)",
                 year, metrics["n_nodes"], metrics["n_edges"])

    rec = {"year": year}
    rec.update(metrics)
    return rec



# =========================
#  데이터셋 처리 루프
# =========================

def process_dataset_fast_parallel(
    df: pd.DataFrame,
    fuel_type: str,
    out_dir: str,
    radius_km: float = 30.0,
    k_min: int = 3,
    r_max_km: Optional[float] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:

    os.makedirs(out_dir, exist_ok=True)

    years_unique = sorted(df["year"].unique())
    if start_year is not None:
        years_unique = [y for y in years_unique if y >= start_year]
    if end_year is not None:
        years_unique = [y for y in years_unique if y <= end_year]

    if not years_unique:
        logging.warning("[%s] No years to process after filtering.", fuel_type)
        return pd.DataFrame()

    logging.info(
        "[%s] Years to process: %s (total %d years)",
        fuel_type,
        years_unique,
        len(years_unique),
    )

    coords_all = df[["lat", "lon"]].values
    years_all = df["year"].values

    # 1) 전역 이웃 한 번만 계산
    radius_dists_list, radius_idx_list, knn_dists, knn_idx = precompute_neighbors(
        coords_all, radius_km=radius_km, k_min=k_min
    )

    # 2) 멀티프로세싱 설정
    if n_jobs is None or n_jobs <= 0:
        n_jobs = mp.cpu_count()
    logging.info("[%s] Using %d parallel workers", fuel_type, n_jobs)

    with mp.Pool(
        processes=n_jobs,
        initializer=_init_worker_year_metrics,
        initargs=(
            years_all,
            coords_all,
            radius_dists_list,
            radius_idx_list,
            knn_dists,
            knn_idx,
            radius_km,
            k_min,
            r_max_km,
        ),
    ) as pool:
        # 연도 리스트를 워커에게 분배
        results = pool.map(_worker_compute_year_metrics, years_unique)

    # fuel_type 정보 붙이기
    for rec in results:
        rec["fuel_type"] = fuel_type

    stats_df = pd.DataFrame(results)
    # year 기준으로 정렬
    stats_df = stats_df.sort_values("year").reset_index(drop=True)

    # 진행상황 로그 예시 하나 찍어주기
    logging.info(
        "[%s] Done. Example row: %s",
        fuel_type,
        stats_df.iloc[-1].to_dict() if not stats_df.empty else "EMPTY",
    )
    return stats_df


# =========================
#  메인
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EV/H2 충전소 네트워크 지표 분석 (고속 버전, 시각화 없음)"
    )
    parser.add_argument("--ev-csv", required=True, help="EV CSV 경로")
    parser.add_argument("--h2-csv", required=True, help="H2 CSV 경로")
    parser.add_argument("--out-dir", required=True, help="출력 디렉토리")
    parser.add_argument("--radius-km", type=float, default=30.0,
                        help="서비스 반경 (km), 기본=30")
    parser.add_argument("--k-min", type=int, default=3,
                        help="하이브리드 최소 degree k_min, 기본=3")
    parser.add_argument("--r-max-km", type=float, default=None,
                        help="kNN 보정 시 최대 허용 거리 (km), 기본=None=제한 없음")
    parser.add_argument("--start-year", type=int, default=None,
                        help="분석 시작 연도 (옵션)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="분석 종료 연도 (옵션)")
    parser.add_argument("--log-level", default="INFO",
                        help="로그 레벨 (DEBUG, INFO, WARNING, ...)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="병렬 워커 수 (기본: CPU 코어 수)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 데이터 로드
    logging.info("Loading EV data...")
    ev_df = load_ev_data(args.ev_csv)

    logging.info("Loading H2 data...")
    h2_df = load_h2_data(args.h2_csv)

    # 2) EV 처리
    logging.info("Processing EV metrics (fast + parallel)...")
    ev_stats = process_dataset_fast_parallel(
        ev_df,
        fuel_type="EV",
        out_dir=args.out_dir,
        radius_km=args.radius_km,
        k_min=args.k_min,
        r_max_km=args.r_max_km,
        start_year=args.start_year,
        end_year=args.end_year,
        n_jobs=args.n_jobs,
    )
    ev_stats_path = os.path.join(args.out_dir, "ev_stats.csv")
    if not ev_stats.empty:
        ev_stats.to_csv(ev_stats_path, index=False)
        logging.info("Saved EV stats CSV: %s", ev_stats_path)
    else:
        logging.warning("EV stats is empty. CSV not saved.")

    # 3) H2 처리
    logging.info("Processing H2 metrics (fast + parallel)...")
    h2_stats = process_dataset_fast_parallel(
        h2_df,
        fuel_type="H2",
        out_dir=args.out_dir,
        radius_km=args.radius_km,
        k_min=args.k_min,
        r_max_km=args.r_max_km,
        start_year=args.start_year,
        end_year=args.end_year,
        n_jobs=args.n_jobs,
    )
    h2_stats_path = os.path.join(args.out_dir, "h2_stats.csv")
    if not h2_stats.empty:
        h2_stats.to_csv(h2_stats_path, index=False)
        logging.info("Saved H2 stats CSV: %s", h2_stats_path)
    else:
        logging.warning("H2 stats is empty. CSV not saved.")


if __name__ == "__main__":
    main()


# python3 analysis.py --ev-csv EVdata.csv --h2-csv H2data.csv --out-dir output2 --radius-km 30 --k-min 3  --n-jobs 1 --log-level INFO
# python3 analysis.py --ev-csv EVdata.csv --h2-csv H2data.csv --out-dir output2 --radius-km 30 --k-min 3  --n-jobs 1 --log-level INFO