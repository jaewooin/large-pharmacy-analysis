from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error


# 파이썬 파일과 같은 폴더에 둘 엑셀 파일
INPUT_FILE = r"C:\Users\jaewoo\Desktop\데이터 수도권 정리.xlsx"
OUTPUT_FILE = "대형약국수_변수별_가중치.csv"

TARGET = "대형약국수"

# 대형약국 수를 예측할 때 사용할 설명 변수
FEATURES = [
    "평균공시지가",
    "중앙값공시지가",
    "총 인구수",
    "연령구간인구수",
    "30~39세",
    "40~49세",
    "50~59세",
    "60~69세",
    "70~79세",
    "80~89세",
    "90~99세",
    "100세 이상",
    "대규모점포수",
]


def to_number(series):
    """'137,048'처럼 쉼표가 들어간 값도 숫자로 변환"""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )


def main():
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / INPUT_FILE
    output_path = base_dir / OUTPUT_FILE

    if not input_path.exists():
        raise FileNotFoundError(
            f"엑셀 파일을 찾을 수 없습니다:\n{input_path}\n"
            "엑셀 파일과 이 파이썬 파일을 같은 폴더에 두세요."
        )

    df = pd.read_excel(input_path)

    required_columns = [TARGET] + FEATURES
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            "엑셀에서 다음 열을 찾지 못했습니다: "
            + ", ".join(missing_columns)
        )

    # 숫자형으로 변환
    for col in required_columns:
        df[col] = to_number(df[col])

    # 목표값이 없는 행은 분석에서 제외
    df = df.dropna(subset=[TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET]

    # 결측값은 각 학습 데이터의 중앙값으로 채우고 랜덤포레스트 학습
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

    # 5겹 교차검증을 10회 반복해 변수 중요도의 변동을 확인
    cv = RepeatedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    fold_importances = []
    fold_mae = []

    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        fold_mae.append(mean_absolute_error(y_test, predictions))

        # 해당 변수를 섞었을 때 예측 성능이 얼마나 나빠지는지 측정
        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        fold_importances.append(result.importances_mean)

    importance_array = np.vstack(fold_importances)

    # 교차검증에서 측정한 변수 중요도의 평균과 변동 폭
    importance_mean = importance_array.mean(axis=0)
    importance_std = importance_array.std(axis=0)

    # 중요도가 음수로 나온 변수는 0으로 처리한 뒤 합계 100%로 환산
    positive_importance = np.maximum(importance_mean, 0)
    total_importance = positive_importance.sum()

    if total_importance == 0:
        weights = np.zeros(len(FEATURES))
        print("주의: 변수 중요도가 모두 0 이하라 가중치를 산출하지 못했습니다.")
    else:
        weights = positive_importance / total_importance * 100

    # 영향 방향 참고용: 각 변수와 대형약국 수의 단순 Spearman 상관
    spearman_corr = X.corrwith(y, method="spearman")

    result_df = pd.DataFrame({
        "변수": FEATURES,
        "교차검증_중요도평균": importance_mean,
        "중요도_변동폭": importance_std,
        "가중치(%)": weights,
        "단순상관계수(방향참고)": [
            spearman_corr.get(col, np.nan) for col in FEATURES
        ]
    })

    result_df = result_df.sort_values(
        "가중치(%)",
        ascending=False
    ).reset_index(drop=True)

    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n[변수별 상대 가중치]")
    print(result_df.to_string(index=False, formatters={
        "교차검증_중요도평균": "{:.4f}".format,
        "중요도_변동폭": "{:.4f}".format,
        "가중치(%)": "{:.2f}".format,
        "단순상관계수(방향참고)": "{:.3f}".format
    }))

    print(f"\n교차검증 평균 MAE: {np.mean(fold_mae):.2f}개")
    print(f"결과 파일 저장 위치: {output_path}")


if __name__ == "__main__":
    main()