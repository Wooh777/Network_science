import pandas as pd
import numpy as np

def preprocess_ev_for_h2(input_path, output_path):
    # 1) 원본 로드
    df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)
    print("원본 행 개수:", len(df))
    print("컬럼:", df.columns.tolist())

    # 2) '위도' 컬럼은 "lat,lon" 형태의 문자열 → lat / lon 분리
    coord_split = df["위도"].astype(str).str.split(",", n=1, expand=True)
    df["lat"] = pd.to_numeric(coord_split[0], errors="coerce")
    df["lon"] = pd.to_numeric(coord_split[1], errors="coerce")

    print("\nlat/lon 생성 후 예시:")
    print(df[["설치년도","시도","군구","주소","충전소명","lat","lon"]].head())

    if '주소' in df.columns:
        before_dup = len(df)
        df = df.drop_duplicates(subset=['주소'], keep='first').copy()
        print(f"주소 기준 중복 제거: {before_dup} → {len(df)}")

    # 3) 위경도 NaN 제거
    before_nonan = len(df)
    df = df.dropna(subset=["lat", "lon"]).copy()
    print(f"\n위경도 NaN 제거: {before_nonan} → {len(df)}")

    # 4) 위경도 범위 기반 이상치 탐지 (한국 대략 경계)
    lat_min, lat_max = 32.0, 39.5
    lon_min, lon_max = 124.0, 132.5

    coord_outlier_mask = (
        (df["lat"] < lat_min) | (df["lat"] > lat_max) |
        (df["lon"] < lon_min) | (df["lon"] > lon_max)
    )
    coord_outliers = df[coord_outlier_mask].copy()
    print("\n[위경도 이상치 개수]:", len(coord_outliers))
    if len(coord_outliers) > 0:
        print(coord_outliers[["설치년도","시도","군구","주소","충전소명","lat","lon"]].head(20))

    # 5) 수소충전소 설치 '불가능 장소' 키워드 정의
    invalid_keywords = [
        "아파트", "APT", "오피스텔", "원룸", "빌라", "기숙사",
        "주차장", "지하", "지하주차장",
        "마트", "백화점", "쇼핑몰", "몰", "아웃렛",
    ]

    def contains_any(text, keywords):
        if pd.isna(text):
            return False
        t = str(text).lower()
        return any(k.lower() in t for k in keywords)

    # 6) 시설구분(소), 충전소명, 주소에 대해 불가능 장소 플래그
    forbidden_mask = (
        df["시설구분(소)"].apply(lambda x: contains_any(x, invalid_keywords)) |
        df["충전소명"].apply(lambda x: contains_any(x, invalid_keywords)) |
        df["주소"].apply(lambda x: contains_any(x, invalid_keywords))
    )

    print("\n[수소충전소 설치 불가능 장소 추정 개수]:", forbidden_mask.sum())
    print("예시 10개:")
    print(df.loc[forbidden_mask, ["시설구분(대)","시설구분(소)","충전소명","주소"]].head(10))

    # 7) 전처리된 데이터셋 생성
    clean_df = df[~coord_outlier_mask & ~forbidden_mask].copy()
    print("\n전처리 후 남은 행 수:", len(clean_df))

    # 8) 저장
    clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n전처리 결과 저장 완료: {output_path}")


if __name__ == "__main__":
    preprocess_ev_for_h2(
        "한국환경공단_전기차 충전소 위치 및 운영 정보_20240708.csv",
        "EVdata.csv",
    )
