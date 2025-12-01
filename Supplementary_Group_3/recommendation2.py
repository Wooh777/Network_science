import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import logging
import random
import math
import argparse
import os
from tqdm import tqdm

EARTH_RADIUS_KM = 6371.0

def haversine_km_vec(lat1_rad, lon1_rad, lat2_rad, lon2_rad):
    """
    lat1_rad, lon1_rad: 기존 노드들 (배열, radian)
    lat2_rad, lon2_rad: 후보 노드 (스칼라, radian)
    return: 거리 (km) 배열
    """
    dlat = lat1_rad - lat2_rad
    dlon = lon1_rad - lon2_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def init_neighbor_counts_fast(h2_coords_deg: np.ndarray, radius_km: float, min_neighbors: int = 3):
    """
    초기 H2 네트워크에 대해:
      - neighbor_counts: 각 노드의 반경 radius_km 이내 이웃 수 (자기 자신 제외)
      - covered_mask: neighbor_counts >= min_neighbors
      - coverage, covered_count, n 반환
    """
    lat_rad = np.radians(h2_coords_deg[:, 0])
    lon_rad = np.radians(h2_coords_deg[:, 1])

    n = lat_rad.shape[0]
    neighbor_counts = np.zeros(n, dtype=np.int32)

    for i in range(n):
        dists_km = haversine_km_vec(lat_rad, lon_rad, lat_rad[i], lon_rad[i])
        neighbor_counts[i] = np.sum(dists_km <= radius_km) - 1  # 자기 자신 제외

    covered_mask = neighbor_counts >= min_neighbors
    covered_count = covered_mask.sum()
    coverage = covered_count / n

    return lat_rad, lon_rad, neighbor_counts, covered_mask, covered_count, coverage

# ---------------------------------------------------------
# HYBRID GRAPH (radius + kNN with max-distance cutoff)
# ---------------------------------------------------------
def build_graph(coords, radius_km=30.0, k_min=3, r_max_km=50.0):
    n = coords.shape[0]
    coords_rad = np.radians(coords)

    G = [[] for _ in range(n)]

    # radius-based neighbors
    nbrs_radius = NearestNeighbors(
        radius=radius_km / EARTH_RADIUS_KM,
        metric="haversine"
    ).fit(coords_rad)
    dist_list, idx_list = nbrs_radius.radius_neighbors(coords_rad)

    for i in range(n):
        for j, d_rad in zip(idx_list[i], dist_list[i]):
            if i == j:
                continue
            dist_km = d_rad * EARTH_RADIUS_KM
            if dist_km <= r_max_km:
                G[i].append(j)
                G[j].append(i)

    # kNN
    k_for_knn = min(k_min + 1, n)
    nbrs_knn = NearestNeighbors(
        n_neighbors=k_for_knn,
        metric="haversine"
    ).fit(coords_rad)

    dist_knn, idx_knn = nbrs_knn.kneighbors(coords_rad)

    for i in range(n):
        if len(G[i]) >= k_min:
            continue

        for j, d_rad in zip(idx_knn[i], dist_knn[i]):
            if i == j:
                continue
            dist_km = d_rad * EARTH_RADIUS_KM
            if dist_km <= r_max_km:
                if j not in G[i]:
                    G[i].append(j)
                    G[j].append(i)
            if len(G[i]) >= k_min:
                break

    # remove duplicates
    G = [list(set(neigh)) for neigh in G]
    return G


# ---------------------------------------------------------
# Metrics (coverage = neighbors>=3, isolated count, gcc_ratio)
# ---------------------------------------------------------
def compute_metrics(coords, radius_km=30.0):
    n = coords.shape[0]
    coords_rad = np.radians(coords)

    nbrs = NearestNeighbors(
        radius=radius_km / EARTH_RADIUS_KM,
        metric="haversine"
    ).fit(coords_rad)

    dist_list, idx_list = nbrs.radius_neighbors(coords_rad)

    # coverage : neighbors >= 3
    isolated = 0
    for idx in idx_list:
        neighbor_cnt = len(idx) - 1
        if neighbor_cnt < 3:
            isolated += 1
    coverage = 1.0 - isolated / n

    # gcc_ratio using hybrid graph
    G = build_graph(coords, radius_km=radius_km, k_min=3, r_max_km=50.0)

    visited = [False] * n

    def dfs(start):
        stack = [start]
        cnt = 0
        while stack:
            u = stack.pop()
            if visited[u]:
                continue
            visited[u] = True
            cnt += 1
            for v in G[u]:
                if not visited[v]:
                    stack.append(v)
        return cnt

    gcc = 0
    for i in range(n):
        if not visited[i]:
            gcc = max(gcc, dfs(i))

    gcc_ratio = gcc / n
    return coverage, isolated, gcc_ratio


# ---------------------------------------------------------
# Candidate generation near EV positions (within 500m)
# ---------------------------------------------------------
def generate_candidates_near_ev(ev_coords, h2_coords,
                                R_meter=500, num_per_ev=1):
    candidates = []

    for (lat, lon) in ev_coords:
        for _ in range(num_per_ev):
            angle = random.uniform(0, 2 * math.pi)
            dist_km = random.uniform(0, R_meter / 1000.0)

            dlat = (dist_km / EARTH_RADIUS_KM) * (180.0 / math.pi)
            dlon = (dist_km / (EARTH_RADIUS_KM * math.cos(math.radians(lat)))) * (180.0 / math.pi)

            new_lat = lat + dlat * math.sin(angle)
            new_lon = lon + dlon * math.cos(angle)
            candidates.append([new_lat, new_lon])

    return np.array(candidates)


# ---------------------------------------------------------
# GREEDY: install minimal stations until coverage ≥ target
# ---------------------------------------------------------
def greedy_h2_until_target(ev_coords,
                           h2_coords_init,
                           coverage_target=0.99,
                           radius_km=30.0,
                           k_min=3,            # 지금은 안 쓰지만 시그니처는 그대로 둠
                           r_max_km=50.0,      # 지금은 안 씀
                           candidate_R_meter=500,
                           num_candidates_per_ev=2):
    """
    Δcoverage 기준 greedy는 그대로 유지하되,
    compute_metrics를 매 후보마다 호출하지 않고
    neighbor_counts를 들고 다니면서 Δcoverage만 빠르게 계산하는 버전.
    """

    # 0) 초기 상태 설정 (H2만)
    if h2_coords_init.shape[0] == 0:
        raise ValueError("초기 H2 좌표가 없습니다.")

    # 초기 neighbor_counts, covered_mask, coverage 계산 (한 번만)
    h2_lat_rad, h2_lon_rad, neighbor_counts, covered_mask, covered_count, coverage = \
        init_neighbor_counts_fast(h2_coords_init, radius_km, min_neighbors=3)

    n = neighbor_counts.shape[0]

    logging.info("Initial H2: N=%d, coverage=%.4f (min_neighbors=3)",
                 n, coverage)

    # history 기록
    history = []
    history.append({
        "step": 0,
        "added_lat": None,
        "added_lon": None,
        "coverage": coverage,
        "isolated": int((neighbor_counts < 3).sum()),
        "gcc_ratio": None,          # 여기서는 안 쓰고 나중에 한 번에 계산해도 됨
        "delta_coverage": 0.0,
    })

    if coverage >= coverage_target:
        logging.info("Already meets coverage target: %.4f >= %.4f",
                     coverage, coverage_target)
        # coords_current는 그냥 초기 H2만
        return h2_coords_init, pd.DataFrame(history)

    # 1) EV 기반 후보 생성 (기존 코드 그대로 사용)
    logging.info("Generating EV-near candidates…")
    candidates_deg = generate_candidates_near_ev(
        ev_coords, h2_coords_init,
        R_meter=candidate_R_meter,
        num_per_ev=num_candidates_per_ev,
    )
    logging.info("Total candidates = %d", len(candidates_deg))

    # 후보를 radian으로도 미리 변환
    cand_lat_rad = np.radians(candidates_deg[:, 0])
    cand_lon_rad = np.radians(candidates_deg[:, 1])

    used = set()
    step = 1

    # 2) greedy 루프 (coverage_target 도달할 때까지 반복)
    while True:
        if coverage >= coverage_target:
            logging.info("Reached target coverage: %.4f (target=%.4f)",
                         coverage, coverage_target)
            break

        if len(used) == len(candidates_deg):
            logging.info("All candidates used up. Stop.")
            break

        best_idx = None
        best_delta = -1e9
        best_cov_new = coverage

        # 현재 상태 캐시
        curr_coverage = coverage
        curr_covered_count = covered_count
        curr_n = n

        radius = radius_km

        # 후보 전부에 대해 Δcoverage 계산 (incremental)
        for i in range(len(candidates_deg)):
            if i in used:
                continue

            clat_rad = cand_lat_rad[i]
            clon_rad = cand_lon_rad[i]

            # 기존 H2들과의 거리 (km)
            dists_km = haversine_km_vec(h2_lat_rad, h2_lon_rad, clat_rad, clon_rad)
            affected = dists_km <= radius

            # 기존 노드 중 neighbor_count == 2였는데 이 후보로 인해 3이 되는 노드 수
            newly_covered_existing = np.logical_and(affected, neighbor_counts == 2)
            num_newly_covered_existing = newly_covered_existing.sum()

            # 새 노드(후보)의 neighbor_count
            cand_neighbors = affected.sum()
            new_node_covered = 1 if cand_neighbors >= 3 else 0

            # 새 상태에서의 covered 개수, coverage 계산
            covered_new = curr_covered_count + num_newly_covered_existing + new_node_covered
            coverage_new = covered_new / (curr_n + 1)
            delta_cov = coverage_new - curr_coverage

            if delta_cov > best_delta:
                best_delta = delta_cov
                best_idx = i
                best_cov_new = coverage_new

        if best_idx is None or best_delta <= 0:
            logging.info("No candidate can improve coverage (Δcov <= 0). Stop.")
            break

        # 3) best 후보 실제로 설치 → 상태 업데이트
        chosen_deg = candidates_deg[best_idx]
        chosen_lat_deg, chosen_lon_deg = float(chosen_deg[0]), float(chosen_deg[1])
        chosen_lat_rad, chosen_lon_rad = cand_lat_rad[best_idx], cand_lon_rad[best_idx]

        used.add(best_idx)

        # 기존 노드들에 대한 neighbor_counts 업데이트
        dists_km_new = haversine_km_vec(h2_lat_rad, h2_lon_rad, chosen_lat_rad, chosen_lon_rad)
        affected_nodes = dists_km_new <= radius
        neighbor_counts[affected_nodes] += 1

        # 새 노드의 neighbor_count
        new_node_neighbor_count = affected_nodes.sum()
        neighbor_counts = np.append(neighbor_counts, new_node_neighbor_count)

        # 좌표(rad, deg 모두) 업데이트
        h2_lat_rad = np.append(h2_lat_rad, chosen_lat_rad)
        h2_lon_rad = np.append(h2_lon_rad, chosen_lon_rad)
        h2_coords_init = np.vstack([h2_coords_init, [chosen_lat_deg, chosen_lon_deg]])

        # covered / coverage 갱신
        covered_mask = neighbor_counts >= 3
        covered_count = covered_mask.sum()
        n = neighbor_counts.shape[0]
        coverage = covered_count / n

        logging.info(
            "STEP %d: add (%.5f, %.5f), Δcov=%.5f → coverage=%.4f",
            step, chosen_lat_deg, chosen_lon_deg, best_delta, coverage
        )

        history.append({
            "step": step,
            "added_lat": chosen_lat_deg,
            "added_lon": chosen_lon_deg,
            "coverage": coverage,
            "isolated": int((neighbor_counts < 3).sum()),
            "gcc_ratio": None,       # 원하면 마지막에 한 번만 compute_metrics로 채워도 됨
            "delta_coverage": best_delta,
        })

        step += 1

    # 여기서 원하면 final_coords에 대해 한 번만 compute_metrics 호출해서
    # 마지막 행의 gcc_ratio 채울 수도 있음.
    return h2_coords_init, pd.DataFrame(history)




def load_ev_coords(ev_csv: str, year: int) -> np.ndarray:
    """
    EV CSV 형식:
      설치년도, ..., lat, lon
    에서 year 이하만 뽑아서 (lat, lon) numpy 배열로 반환
    """
    df = pd.read_csv(ev_csv)
    if "설치년도" not in df.columns:
        raise ValueError("EV CSV에 '설치년도' 컬럼이 없습니다.")

    df_year = df[df["설치년도"] <= year].copy()
    if not {"lat", "lon"}.issubset(df_year.columns):
        raise ValueError("EV CSV에 'lat', 'lon' 컬럼이 필요합니다.")

    coords = df_year[["lat", "lon"]].values
    logging.info("Loaded EV: %d rows (<= %d년)", coords.shape[0], year)
    return coords


def load_h2_coords(h2_csv: str, year: int) -> np.ndarray:
    """
    H2 CSV 형식:
      station_id, year, name, 주소, 구분, 공급방식, lat, lon
    에서 year 이하만 뽑아서 (lat, lon) numpy 배열로 반환
    """
    df = pd.read_csv(h2_csv)
    if "year" not in df.columns:
        raise ValueError("H2 CSV에 'year' 컬럼이 없습니다.")

    df_year = df[df["year"] <= year].copy()
    if not {"lat", "lon"}.issubset(df_year.columns):
        raise ValueError("H2 CSV에 'lat', 'lon' 컬럼이 필요합니다.")

    coords = df_year[["lat", "lon"]].values
    logging.info("Loaded H2: %d rows (<= %d년)", coords.shape[0], year)
    return coords


def parse_args():
    p = argparse.ArgumentParser(
        description="H2 신규 충전소 최적 설치 (coverage 목표까지 자동 설치)"
    )

    p.add_argument("--ev-csv", required=True)
    p.add_argument("--h2-csv", required=True)
    # p.add_argument("--year", type=int, default=2024)
    p.add_argument("--coverage-target", type=float, default=0.99)

    # 네트워크 파라미터
    p.add_argument("--radius-km", type=float, default=30.0)
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--r-max-km", type=float, default=50.0)

    # 후보 생성
    p.add_argument("--candidate-radius-m", type=float, default=500)
    p.add_argument("--num-candidates-per-ev", type=int, default=1)

    p.add_argument("--out-prefix", required=True)
    p.add_argument("--log-level", default="INFO")

    return p


def main():
    args = parse_args().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s"
    )

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)

    # Load EV / H2
    ev_coords = load_ev_coords(args.ev_csv, 2024)
    h2_coords = load_h2_coords(args.h2_csv, 2025)

    # Greedy optimization (no limit)
    final_coords, history = greedy_h2_until_target(
        ev_coords=ev_coords,
        h2_coords_init=h2_coords,
        coverage_target=args.coverage_target,
        radius_km=args.radius_km,
        k_min=args.k_min,
        r_max_km=args.r_max_km,
        candidate_R_meter=args.candidate_radius_m,
        num_candidates_per_ev=args.num_candidates_per_ev
    )

    # Save outputs
    hist_path = args.out_prefix + "_history.csv"
    final_path = args.out_prefix + "_coords.csv"

    history.to_csv(hist_path, index=False)
    pd.DataFrame(final_coords, columns=["lat", "lon"]).to_csv(final_path, index=False)

    logging.info("Saved history: %s", hist_path)
    logging.info("Saved final coords: %s", final_path)

if __name__ == "__main__":
    main()

# python3 recommendation2.py --ev-csv EVdata.csv --h2-csv H2data.csv --coverage-target 0.99 --out-prefix output/H2_optimal_2024_2 --log-level INFO