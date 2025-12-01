#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Failure Scenario Analysis for EV/H2 Charging Networks

- 특정 연도까지 누적된 네트워크(하이브리드: radius 30km + k_min, r_max=50km)에 대해
- 노드를 하나씩 제거했을 때
    Δcoverage, Δisolated, Δgcc
  변화량을 계산하여 CSV + 그래프를 출력한다.

⚠️ 노드 수가 많은 EV(수만 개)에 그대로 돌리면 매우 느릴 수 있음.
   → H2에는 full로 써도 되고,
     EV는 특정 지역 subset / 샘플링해서 쓰는 것을 추천.
"""

import argparse
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

EARTH_RADIUS_KM = 6371.0


# ======================
#  데이터 로드
# ======================

def load_data(csv_path: str, fuel_type: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if fuel_type == "EV":
        # EV: 설치년도 컬럼명을 year로 통일
        if "설치년도" not in df.columns:
            raise ValueError("EV CSV에 '설치년도' 컬럼이 없습니다.")
        df = df.rename(columns={"설치년도": "year"})
    else:
        # H2: year 컬럼 있다고 가정
        if "year" not in df.columns:
            raise ValueError("H2 CSV에 'year' 컬럼이 없습니다.")

    for col in ["lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"CSV에 '{col}' 컬럼이 없습니다.")

    out = df[["year", "lat", "lon"]].copy()
    out = out.dropna(subset=["year", "lat", "lon"])
    out["year"] = out["year"].astype(int)
    out = out.sort_values("year").reset_index(drop=True)

    logging.info(
        "Loaded %s data: %d rows, year range = [%s, %s]",
        fuel_type, len(out),
        out["year"].min(), out["year"].max()
    )
    return out


# ======================
#  그래프 구성
# ======================

def build_graph(
    coords: np.ndarray,
    radius_km: float = 30.0,
    k_min: int = 3,
    r_max_km: float = 50.0,
) -> nx.Graph:
    """
    coords: (N, 2) = [lat, lon]
    하이브리드 규칙:
      - radius 30km 안 이웃은 다 연결
      - 추가로 degree < k_min 인 노드는
        kNN으로 채우되, d <= r_max_km (50km)까지만 연결
    """
    n = coords.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    if n == 0:
        return G

    coords_rad = np.radians(coords)

    # 1) radius 이웃
    nbrs_radius = NearestNeighbors(
        radius=radius_km / EARTH_RADIUS_KM,
        metric="haversine"
    ).fit(coords_rad)
    dist_list, idx_list = nbrs_radius.radius_neighbors(coords_rad)

    for i in range(n):
        dists_i = dist_list[i]
        idx_i = idx_list[i]
        for d_rad, j in zip(dists_i, idx_i):
            if i == j:
                continue
            d_km = d_rad * EARTH_RADIUS_KM
            if d_km <= r_max_km:
                G.add_edge(i, j, weight=d_km)

    # 2) kNN 보정
    k_for_knn = min(k_min + 1, n)
    nbrs_knn = NearestNeighbors(
        n_neighbors=k_for_knn,
        metric="haversine"
    ).fit(coords_rad)
    knn_dists, knn_idx = nbrs_knn.kneighbors(coords_rad)

    for i in range(n):
        if G.degree[i] >= k_min:
            continue

        d_knn_i = knn_dists[i][1:]   # 자기 자신 제외
        idx_knn_i = knn_idx[i][1:]

        current_deg = G.degree[i]
        for d_rad, j in zip(d_knn_i, idx_knn_i):
            d_km = d_rad * EARTH_RADIUS_KM
            if d_km > r_max_km:
                continue
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=d_km)
                current_deg += 1
                if current_deg >= k_min:
                    break

    return G


# ======================
#  지표 계산
# ======================

def compute_metrics(
    G: nx.Graph,
    coords: np.ndarray,
    radius_km: float = 30.0,
) -> Tuple[float, int, float]:
    """
    coverage_ratio, n_isolated_R, gcc_ratio 반환
    """
    n = len(G.nodes())
    if n == 0:
        return 0.0, 0, 0.0

    coords_rad = np.radians(coords)
    nbrs_radius = NearestNeighbors(
        radius=radius_km / EARTH_RADIUS_KM,
        metric="haversine"
    ).fit(coords_rad)
    dist_list, idx_list = nbrs_radius.radius_neighbors(coords_rad)

    # isolated = 0
    # for idx in idx_list:
    #     # 자기 자신 제외
    #     if len(idx) - 1 == 0:
    #         isolated += 1

    isolated = 0
    for idx in idx_list:
        neighbor_count = len(idx) - 1   # 자기 자신 제외
        if neighbor_count < 3:          # 🔥 핵심 변경
            isolated += 1

    coverage_ratio = 1.0 - isolated / n

    comps = list(nx.connected_components(G))
    if len(comps) == 0:
        gcc_ratio = 0.0
    else:
        largest = max(comps, key=len)
        gcc_ratio = len(largest) / n

    return coverage_ratio, isolated, gcc_ratio


# ======================
#  Failure scenario 본체
# ======================

def failure_scenario(
    coords: np.ndarray,
    radius_km: float = 30.0,
    k_min: int = 3,
    r_max_km: float = 50.0,
) -> pd.DataFrame:
    """
    각 노드 i를 하나씩 제거해보면서
    Δcoverage, Δisolated, Δgcc 계산
    """
    logging.info("Building base graph...")
    G_base = build_graph(coords, radius_km=radius_km, k_min=k_min, r_max_km=r_max_km)
    base_cov, base_iso, base_gcc = compute_metrics(G_base, coords, radius_km=radius_km)

    logging.info(
        "Base metrics: coverage=%.4f, isolated=%d, gcc_ratio=%.4f",
        base_cov, base_iso, base_gcc
    )

    n = coords.shape[0]
    results = []

    for i in range(n):
        # 노드 i 제거
        coords_removed = np.delete(coords, i, axis=0)
        G_removed = build_graph(
            coords_removed,
            radius_km=radius_km,
            k_min=k_min,
            r_max_km=r_max_km
        )
        cov, iso, gcc = compute_metrics(G_removed, coords_removed, radius_km=radius_km)

        results.append({
            "node_removed": i,
            "base_coverage": base_cov,
            "after_coverage": cov,
            "delta_coverage": cov - base_cov,
            "base_isolated": base_iso,
            "after_isolated": iso,
            "delta_isolated": iso - base_iso,
            "base_gcc": base_gcc,
            "after_gcc": gcc,
            "delta_gcc": gcc - base_gcc,
            "removed_lat": coords[i, 0],
            "removed_lon": coords[i, 1],
        })

        if (i + 1) % 20 == 0 or i == n - 1:
            logging.info("  processed %d/%d nodes...", i + 1, n)

    df_res = pd.DataFrame(results)
    return df_res

def failure_scenario_cumulative(
    coords,
    radius_km=30.0,
    k_min=3,
    r_max_km=50.0,
    removal_order=None,
):
    """
    누적 제거(Cascading Failure) 시나리오:
      - 처음 N개에서 시작해서
      - 한 번 제거한 상태를 기준으로 계속 다음 노드를 제거해 나감.
      - 각 단계에서 coverage / isolated / gcc 변화를 기록.

    removal_order:
      - 제거할 노드의 '원본 인덱스' 순서 리스트 (예: [0, 5, 10, ...])
      - None이면 [0, 1, 2, ..., N-1] 순서로 제거
    """
    n = coords.shape[0]
    if n == 0:
        return pd.DataFrame()

    # 제거 순서 설정 (원본 인덱스 기준)
    if removal_order is None:
        removal_order = np.arange(n)
    else:
        removal_order = np.array(removal_order)

    # 현재 살아 있는 노드들의 "원본 인덱스"를 추적
    remaining_idx = np.arange(n)          # 길이 = 현재 남은 노드 수
    coords_current = coords.copy()

    # 초기 전체 그래프 및 지표
    logging.info("Building base graph (cumulative failure)...")
    G_current = build_graph(
        coords_current,
        radius_km=radius_km,
        k_min=k_min,
        r_max_km=r_max_km,
    )
    cov_current, iso_current, gcc_current = compute_metrics(
        G_current,
        coords_current,
        radius_km=radius_km,
    )

    base_cov = cov_current
    base_iso = iso_current
    base_gcc = gcc_current

    logging.info(
        "Base metrics (step=0): coverage=%.4f, isolated=%d, gcc_ratio=%.4f",
        base_cov, base_iso, base_gcc
    )

    results = []

    # (선택) step=0 상태도 기록하고 싶으면 여기에 append 해도 됨
    results.append({
        "step": 0,
        "removed_original_index": -1,   # 아직 제거 없음
        "n_nodes_before": coords_current.shape[0],
        "n_nodes_after": coords_current.shape[0],
        "coverage_before": cov_current,
        "coverage_after": cov_current,
        "delta_coverage": 0.0,
        "isolated_before": iso_current,
        "isolated_after": iso_current,
        "delta_isolated": 0,
        "gcc_before": gcc_current,
        "gcc_after": gcc_current,
        "delta_gcc": 0.0,
        "removed_lat": np.nan,
        "removed_lon": np.nan,
    })

    # 누적 제거 루프
    step = 1
    for orig_id in removal_order:
        if coords_current.shape[0] <= 1:
            # 노드 1개 이하면 더 이상 의미 있는 네트워크 아님
            break

        # 아직 남아있는 노드 중에 이 orig_id가 있는지 확인
        pos = np.where(remaining_idx == orig_id)[0]
        if len(pos) == 0:
            # 이미 제거된 노드라면 스킵
            continue
        pos = pos[0]

        # 제거 전 상태 기록
        cov_before, iso_before, gcc_before = cov_current, iso_current, gcc_current
        n_before = coords_current.shape[0]

        removed_lat = coords_current[pos, 0]
        removed_lon = coords_current[pos, 1]

        # 해당 노드 제거
        coords_next = np.delete(coords_current, pos, axis=0)
        remaining_next = np.delete(remaining_idx, pos)

        # 새 그래프 구성
        G_next = build_graph(
            coords_next,
            radius_km=radius_km,
            k_min=k_min,
            r_max_km=r_max_km,
        )
        cov_next, iso_next, gcc_next = compute_metrics(
            G_next,
            coords_next,
            radius_km=radius_km,
        )

        n_after = coords_next.shape[0]

        results.append({
            "step": step,
            "removed_original_index": int(orig_id),
            "n_nodes_before": int(n_before),
            "n_nodes_after": int(n_after),
            "coverage_before": cov_before,
            "coverage_after": cov_next,
            "delta_coverage": cov_next - cov_before,
            "isolated_before": int(iso_before),
            "isolated_after": int(iso_next),
            "delta_isolated": int(iso_next - iso_before),
            "gcc_before": gcc_before,
            "gcc_after": gcc_next,
            "delta_gcc": gcc_next - gcc_before,
            "removed_lat": float(removed_lat),
            "removed_lon": float(removed_lon),
        })

        # 다음 단계 준비
        coords_current = coords_next
        remaining_idx = remaining_next
        G_current = G_next
        cov_current, iso_current, gcc_current = cov_next, iso_next, gcc_next

        if step % 20 == 0 or step == len(removal_order):
            logging.info(
                "  [step=%d] n=%d, coverage=%.4f, isolated=%d, gcc=%.4f",
                step, n_after, cov_next, iso_next, gcc_next
            )

        step += 1

    df_res = pd.DataFrame(results)
    return df_res


# ======================
#  그래프 그리기
# ======================
def plot_failure_results(
    df_res: pd.DataFrame,
    out_prefix: str,
    fuel_type: str,
    year: int,
):
    # ==========================
    # 1) after_coverage 꺾은선 그래프 (원래 순서)
    # ==========================

    x = df_res["node_removed"].values
    y = df_res["after_coverage"].values
    base_cov = df_res["base_coverage"].iloc[0]

    plt.figure(figsize=(8,4))
    plt.plot(x, y, marker="o", markersize=3, linewidth=1.0)
    plt.axhline(base_cov, linestyle="--", color="red", label="base coverage")

    plt.xlabel("node_removed (index)")
    plt.ylabel("after_coverage")
    plt.title(f"{fuel_type} Failure Scenario (year={year})\nCoverage ratio after removing each node")
    plt.legend()
    plt.tight_layout()

    line_path = f"{out_prefix}_coverage_by_node.png"
    plt.savefig(line_path, dpi=200)
    plt.close()
    logging.info("Saved: %s", line_path)

    # ==========================
    # 2) delta_coverage 라인 그래프
    # ==========================

    deltas = df_res["delta_coverage"].values

    plt.figure(figsize=(8,4))
    plt.plot(x, deltas, marker="o", markersize=3, linewidth=1.0)
    plt.axhline(0, linestyle="--", color="gray")

    plt.xlabel("node_removed (index)")
    plt.ylabel("Δ coverage")
    plt.title(f"{fuel_type} Failure Scenario (year={year})\nΔcoverage per removed node")
    plt.tight_layout()

    delta_path = f"{out_prefix}_delta_coverage_by_node.png"
    plt.savefig(delta_path, dpi=200)
    plt.close()
    logging.info("Saved: %s", delta_path)

def plot_cumulative_coverage(df_res: pd.DataFrame, out_path: str, fuel_type: str, year: int):
    # step > 0만 사용해도 되고, 0 포함해도 됨
    x = df_res["step"].values
    y = df_res["coverage_after"].values

    plt.figure(figsize=(8,4))
    plt.plot(x, y, marker="o", markersize=2, linewidth=1.0)
    plt.xlabel("Step (cumulative node removal)")
    plt.ylabel("Coverage ratio (after)")
    plt.title(f"{fuel_type} cumulative failure (year={year})\nCoverage vs removal step")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ======================
#  main
# ======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Failure Scenario (충전소 고장/폐쇄 시 영향 분석)"
    )
    parser.add_argument("--csv", required=True, help="EV 또는 H2 데이터 CSV 경로")
    parser.add_argument("--fuel-type", choices=["EV", "H2"], required=True,
                        help="연료 타입: EV 또는 H2")
    parser.add_argument("--year", type=int, required=True,
                        help="이 연도까지 누적된 네트워크 기준으로 분석")
    parser.add_argument("--out-prefix", required=True,
                        help="출력 파일 prefix (예: output/H2_failure_2024)")
    parser.add_argument("--radius-km", type=float, default=30.0,
                        help="서비스 반경 (기본=30km)")
    parser.add_argument("--k-min", type=int, default=3,
                        help="최소 차수 보정용 k_min (기본=3)")
    parser.add_argument("--r-max-km", type=float, default=50.0,
                        help="최대 연결 거리 (기본=50km)")
    parser.add_argument("--log-level", default="INFO",
                        help="로그 레벨 (DEBUG, INFO, WARNING, ...)")
    return parser.parse_args()


# def main():
#     args = parse_args()
#     logging.basicConfig(
#         level=getattr(logging, args.log_level.upper(), logging.INFO),
#         format="%(levelname)s:%(message)s"
#     )

#     os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

#     df_all = load_data(args.csv, args.fuel_type)

#     # 해당 연도까지 누적
#     df_year = df_all[df_all["year"] <= args.year].copy()
#     coords = df_year[["lat", "lon"]].values

#     logging.info(
#         "Running failure scenario for %s up to year=%d (N=%d)",
#         args.fuel_type, args.year, len(coords)
#     )

#     df_res = failure_scenario(
#         coords,
#         radius_km=args.radius_km,
#         k_min=args.k_min,
#         r_max_km=args.r_max_km,
#     )

#     # CSV 저장
#     csv_path = f"{args.out_prefix}_results.csv"
#     df_res.to_csv(csv_path, index=False)
#     logging.info("Saved CSV: %s", csv_path)

#     # 그래프 저장
#     plot_failure_results(
#         df_res,
#         out_prefix=args.out_prefix,
#         fuel_type=args.fuel_type,
#         year=args.year,
#     )


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s"
    )

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # 데이터 로드
    df_all = load_data(args.csv, args.fuel_type)

    df_year = df_all[df_all["year"] <= args.year].copy()
    coords = df_year[["lat", "lon"]].values

    logging.info(
        "Running **cumulative** failure scenario for %s up to year=%d (N=%d)",
        args.fuel_type, args.year, len(coords)
    )

    # 여기!! 단일 → 누적 제거로 교체
    df_res = failure_scenario_cumulative(
        coords,
        radius_km=args.radius_km,
        k_min=args.k_min,
        r_max_km=args.r_max_km,
    )

    # CSV 저장
    csv_path = f"{args.out_prefix}_cumulative_results.csv"
    df_res.to_csv(csv_path, index=False)
    logging.info("Saved CSV: %s", csv_path)

    # 그래프 (누적 버전 전용)
    plot_cumulative_coverage(
        df_res,
        out_path=f"{args.out_prefix}_cumulative_coverage.png",
        fuel_type=args.fuel_type,
        year=args.year,
    )


if __name__ == "__main__":
    main()

# python3 failure.py --csv EVdata.csv --fuel-type EV --year 2017 --out-prefix output/EV_failure_2017 --radius-km 30 --k-min 3 --r-max-km 50 --log-level INFO
# python3 failure.py --csv H2data.csv --fuel-type H2 --year 2025 --out-prefix output/H2_failure_2025 --radius-km 30 --k-min 3 --r-max-km 50 --log-level INFO