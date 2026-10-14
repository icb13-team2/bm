"""다운캐스트 parquet 불러오기 헬퍼.

원문 컬럼은 category -> string으로 바꿔서 .str 함수를 바로 쓸 수 있게 해 줘요.
라벨 컬럼(카테고리, 화자, QA, 고객의도, 상담사의도 등)은 category 그대로 둬요.

사용 예 (데이터_다운캐스트/ 폴더에서):
    from src.load_data import load_shopping
    df = load_shopping()
    df["고객질문(요청)"].str.contains("배송").sum()
"""
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path("./data/shopping_02_category_zstd-19.parquet")

TEXT_COLS = ["고객질문(요청)", "상담사질문(요청)", "고객답변", "상담사답변",
             "개체명", "용어사전", "지식베이스"]


def load_shopping(path=DEFAULT_PATH, columns=None, text_as_string=True):
    """parquet을 읽고, 원문 컬럼을 string 타입으로 바꿔서 돌려줘요.

    columns: 필요한 컬럼만 읽고 싶을 때 리스트로 넘겨요.
    text_as_string: False면 category 그대로 둬요(메모리 절약용).
    """
    df = pd.read_parquet(path, columns=columns)
    if text_as_string:
        for c in TEXT_COLS:
            if c in df.columns:
                df[c] = df[c].astype("string")
    return df
