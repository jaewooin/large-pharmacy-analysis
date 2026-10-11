from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline


# 실제 엑셀 파일 위치에 맞게 수정
INPUT_FILE = Path(
    r"C:\Users\jaewoo\Desktop\데이터 수도권 정리.xlsx"
)
OUTPUT_FILE = INPUT_FILE.with_name(
    "인구대비_대형약국수_변수별_가중치.csv"
)

TARGET_COUNT = "대형약국수"
POPULATION = "총 인구수"

AGE_COLUMNS = [
    "30~39세",
    "40~49세",
    "50~59세",
    "60~69세",
    "70~79세",
    "80~89세",
    "90~99세",
    "100세 이상",
]


def to_number(series):
    """쉼표가 들어간 인구수를 숫자로 변환"""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"엑셀 파일을 찾을 수 없습니다:\n{INPUT_FILE}"
        )

    df = pd.read_excel(INPUT_FILE)

    required = [
        TARGET_COUNT,
        POPULATION,
        "중앙값공시지가",
        "대규모점포수",
        *AGE_COLUMNS,
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "엑셀에서 다음 열을 찾지 못했습니다: "
            + ", ".join(missing)
        )

    for col in required:
        df[col] = to_number(df[col])

    # 목표값과 인구가 있어야 계산 가능
    df = df.dropna(subset=[TARGET_COUNT, POPULATION]).copy()
    df = df[df[POPULATION] > 0].copy()

    # 목표값: 인구 1만 명당 대형약국 수
    df["인구1만명당_대형약국수"] = (
        df[TARGET_COUNT] / df[POPULATION] * 10000
    )

    # 연령별 인구 구성비
    age_ratio_columns = []
    for col in AGE_COLUMNS:
        ratio_col = f"{col}_비율(%)"
        df[ratio_col] = df[col] / df[POPULATION] * 100
        age_ratio_columns.append(ratio_col)

    # 대규모점포 수도 인구 1만 명당 수로 변환
    df["인구1만명당_대규모점포수"] = (
        df["대규모점포수"] / df[POPULATION] * 10000
    )

    # 인구 규모와 약국 수 자체는 설명 변수에서 제외
    feature_columns = [
        "중앙값공시지가",
        "인구1만명당_대규모점포수",
        *age_ratio_columns,
    ]

    X = df[feature_columns]
    y = df["인구1만명당_대형약국수"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1
        ))
    ])

    # 5겹 교차검증 5회 반복
    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42
    )

    fold_importances = []
    mae_scores = []

    for train_idx, test_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, predictions))

        # 변수를 섞었을 때 예측 성능이 얼마나 나빠지는지 측정
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=42,
            n_jobs=1
        )

        fold_importances.append(result.importances_mean)

    importance_array = np.vstack(fold_importances)
    importance_mean = importance_array.mean(axis=0)
    importance_std = importance_array.std(axis=0)

    # 양수 중요도만 합계 100%로 환산
    positive_importance = np.maximum(importance_mean, 0)
    total_importance = positive_importance.sum()

    if total_importance > 0:
        weights = positive_importance / total_importance * 100
    else:
        weights = np.zeros(len(feature_columns))
        print("주의: 양수 중요도가 없어 가중치를 산출하지 못했습니다.")

    # 단순 상관은 영향 방향을 참고하는 값
    spearman_corr = X.corrwith(y, method="spearman")

    result_df = pd.DataFrame({
        "변수": feature_columns,
        "중요도평균": importance_mean,
        "중요도변동폭": importance_std,
        "가중치(%)": weights,
        "단순상관계수_방향참고": [
            spearman_corr.get(col, np.nan)
            for col in feature_columns
        ],
    }).sort_values("가중치(%)", ascending=False)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[인구 1만 명당 대형약국 수 기준 가중치]")
    print(result_df.to_string(index=False, formatters={
        "중요도평균": "{:.4f}".format,
        "중요도변동폭": "{:.4f}".format,
        "가중치(%)": "{:.2f}".format,
        "단순상관계수_방향참고": "{:.3f}".format,
    }))

    print(f"\n가중치 합계: {result_df['가중치(%)'].sum():.2f}%")
    print(
        "교차검증 평균 MAE: "
        f"{np.mean(mae_scores):.4f}개/인구 1만 명"
    )
    print(f"결과 파일 저장 위치: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()