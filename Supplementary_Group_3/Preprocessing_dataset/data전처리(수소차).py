import pandas as pd

# 원본 파일 경로
filepath = "한국가스안전공사_수소충전소 현황_20250630.csv"

# CSV 로드
df = pd.read_csv(filepath)

# 위경도 숫자 변환
df["lat"] = pd.to_numeric(df["위도"], errors="coerce")
df["lon"] = pd.to_numeric(df["경도"], errors="coerce")

# NaN 좌표 제거
df = df.dropna(subset=["lat", "lon"])

# 한국 범위 내 좌표만 유지
mask_korea = (
    (df["lat"] >= 32.0) & (df["lat"] <= 39.5) &
    (df["lon"] >= 124.0) & (df["lon"] <= 132.5)
)
df = df[mask_korea]

# 주소 기준 중복 제거
df = df.drop_duplicates(subset=["주소"], keep="first")

# ------------------------------
# ⛔ 상업용만 선택
# ------------------------------
df_commercial = df[df["용도"] == "상업용"].copy()

# 네트워크용 컬럼 맞추기
df_commercial = df_commercial.rename(columns={
    "번호": "station_id",
    "구축연도": "year",
    "충전소명": "name"
})

# year를 숫자로 통일 (YYYY)
df_commercial["year"] = df_commercial["year"].astype(str).str[:4].astype(int)

# 최종 네트워크용 컬럼 정리
df_commercial = df_commercial[
    ["station_id", "year", "name", "주소", "구분", "공급방식", "lat", "lon"]
]

# 저장
output_path = "H2data.csv"
df_commercial.to_csv(output_path, index=False)

print("전처리 완료!")
print(f"총 상업용 수소충전소 개수: {len(df_commercial)}")
print(f"저장 위치: {output_path}")
