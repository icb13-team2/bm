"""tidy(long form) 3개 표로 나눠 저장.

실행 위치: 데이터_다운캐스트/ 폴더
    uv run --no-project --with pandas --with pyarrow --with tabulate python src/make_tidy.py

만드는 표
- tidy_dialogs     : 한 행 = 대화 1건
- tidy_utterances  : 한 행 = 발화 1줄
- tidy_entities    : 한 행 = 발화 안의 개체/지식/용어 값 1개
중간 산출물이라 저장 속도가 빠른 zstd-3으로 저장해요. (원본 base 파일은 zstd-19 그대로)
"""
import time
from pathlib import Path

import pandas as pd

BASE = Path("./data/shopping_02_category_zstd-19.parquet")
OUT = Path("./data")
REPORT = Path("./report")
TEXT_COLS = ["고객질문(요청)", "상담사질문(요청)", "고객답변", "상담사답변"]


def log(m):
    print(time.strftime("%H:%M:%S"), m, flush=True)


def mb(p):
    return p.stat().st_size / 1024 ** 2


def clean_category(s):
    s = s.astype(str).str.strip()
    return s.where(s != "as", "AS")


def describe_cat(t, text_cols=()):
    """범주형 기술 통계. 원문 열의 top 값은 글자수로 가려요."""
    cols = [c for c in t.columns if not pd.api.types.is_numeric_dtype(t[c])]
    d = t[cols].astype(str).describe().T
    d["top"] = [f"({len(x)}자)" if c in text_cols else (x if x else "(빈값)")
                for c, x in zip(d.index, d["top"])]
    return d


def stats_section(dia, utt, ent):
    utt_len = utt["발화"].str.len()
    num_utt = pd.DataFrame({"문장번호": utt["문장번호"], "발화글자수": utt_len}).describe().T.round(2)
    num_dia = dia[["총턴수", "발화수", "고객발화수", "상담사발화수"]].describe().T.round(2)
    pct = dia["총턴수"].quantile([.5, .75, .9, .95, .99]).astype(int).to_dict()
    per_cat = dia.groupby("카테고리", observed=True)["총턴수"].agg(["count", "mean", "median", "max"]).round(2)
    per_cat.columns = ["대화수", "평균턴수", "중앙값", "최대"]
    return f"""
## 기술 통계 (tidy 표 기준)

### 수치형 - tidy_dialogs (대화 1건 = 1행)
{num_dia.to_markdown()}

- 총턴수 분위수: {pct} (50%, 75%, 90%, 95%, 99%)

### 수치형 - tidy_utterances (발화 1줄 = 1행)
{num_utt.to_markdown()}

- `발화글자수`는 원문 길이만 센 값이에요. 원문 내용은 싣지 않았어요.

### 범주형 - tidy_dialogs
{describe_cat(dia).to_markdown()}

### 범주형 - tidy_utterances
{describe_cat(utt, text_cols=("발화",)).to_markdown()}

### 범주형 - tidy_entities
{describe_cat(ent, text_cols=("값",)).to_markdown()}

### 카테고리별 턴수
{per_cat.sort_values("대화수", ascending=False).to_markdown()}

### 라벨 빈도
- 화자: {utt['화자'].value_counts().to_dict()}
- QA: {utt['QA'].value_counts().to_dict()}
- 의도기준: {utt['의도기준'].value_counts().to_dict()}
- 개체 표 출처: {ent['출처'].value_counts().to_dict()}
"""


MARK = "<!-- tidy-section -->"


def update_report2(sizes, n_ent):
    """기존 02 보고서 끝에 tidy 변환 요약 섹션을 넣거나 갱신해요."""
    p = REPORT / "02_category_압축_보고서.md"
    txt = p.read_text(encoding="utf-8")
    txt = txt.split(MARK)[0].rstrip() + "\n"
    sec = f"""
{MARK}
## 8. tidy-data(long form)로 정리

원본 base 파일은 그대로 두고, 분석용으로 3개 표를 따로 만들었어요. 자세한 내용과 기술 통계는 [03_tidy_변환_보고서.md](./03_tidy_변환_보고서.md)에 있어요.

{sizes.drop(columns=["열"]).to_markdown(index=False)}

- 발화문 4개 열 → `발화` 1개, 의도 2개 열 → `의도` 1개로 합쳤어요.
- 카테고리 값의 공백·대소문자를 정리했어요.
- 쉼표로 묶인 값은 `tidy_entities`에 한 줄씩 풀었어요.
- 중간 산출물은 빠른 zstd-3으로 저장해요. 다시 저장할 일이 잦아도 부담이 적어요.
"""
    p.write_text(txt + sec, encoding="utf-8")


def main():
    df = pd.read_parquet(BASE)
    for c in df.columns:
        if str(df[c].dtype) == "category":
            df[c] = df[c].astype(str)
    df["카테고리"] = clean_category(df["카테고리"])
    log(f"로드 완료 {len(df):,}행, 카테고리 {df['카테고리'].nunique()}개")

    # ---- 발화 표 ----
    utt = df[["대화셋일련번호", "문장번호", "화자", "QA"]].copy()
    utt["발화"] = ""
    for c in TEXT_COLS:
        m = df[c] != ""
        utt.loc[m, "발화"] = df.loc[m, c]
    cust, agent = df["고객의도"], df["상담사의도"]
    both = ((cust != "") & (agent != "")).sum()
    # 화자와 같은 쪽 의도 열을 우선 사용: 고객 발화는 고객의도, 상담사 발화는 상담사의도
    own = cust.where(df["화자"] == "고객", agent)
    other = agent.where(df["화자"] == "고객", cust)
    utt["의도"] = own.where(own != "", other)
    utt["보조의도"] = other.where((own != "") & (other != ""), "")  # 두 열이 동시에 있을 때 버려지는 쪽 보존
    utt["의도기준"] = "없음"
    utt.loc[own != "", "의도기준"] = "화자일치열"
    utt.loc[(own == "") & (other != ""), "의도기준"] = "반대열"
    utt["보조의도"] = utt["보조의도"].astype("category")
    utt["화자"] = utt["화자"].astype("category")
    utt["QA"] = utt["QA"].astype("category")
    utt["의도기준"] = utt["의도기준"].astype("category")
    utt["의도"] = utt["의도"].astype("category")
    utt["대화셋일련번호"] = utt["대화셋일련번호"].astype("category")

    # ---- 대화 표 ----
    g = df.groupby("대화셋일련번호", sort=False)
    dia = pd.DataFrame({
        "카테고리": g["카테고리"].first(),
        "총턴수": g["문장번호"].max(),
        "발화수": g.size(),
        "고객발화수": g["화자"].apply(lambda s: (s == "고객").sum()),
        "상담사발화수": g["화자"].apply(lambda s: (s == "상담사").sum()),
    }).reset_index()
    dia["카테고리"] = dia["카테고리"].astype("category")
    dia[["총턴수", "발화수", "고객발화수", "상담사발화수"]] = dia[
        ["총턴수", "발화수", "고객발화수", "상담사발화수"]].apply(pd.to_numeric, downcast="unsigned")
    log("대화·발화 표 완료")

    # ---- 개체 표 (long) ----
    parts = []
    for col in ["개체명", "지식베이스", "용어사전"]:
        s = df[col]
        m = s != ""
        t = df.loc[m, ["대화셋일련번호", "문장번호"]].copy()
        t["출처"] = col
        parts.append((t, s[m]))
    rows = []
    for t, s in parts:
        t = t.assign(값=s.str.split(",")).explode("값")
        t["값"] = t["값"].str.strip()
        t = t[t["값"] != ""]
        if t["출처"].iloc[0] == "개체명":
            t["유형"] = ""
        else:
            sp = t["값"].str.rsplit("/", n=1, expand=True)
            t["값"] = sp[0]
            t["유형"] = sp[1].fillna("") if sp.shape[1] > 1 else ""
        rows.append(t)
    ent = pd.concat(rows, ignore_index=True)
    for c in ["대화셋일련번호", "출처", "유형"]:
        ent[c] = ent[c].astype("category")
    ent["값"] = ent["값"].astype("category")
    log("개체 표 완료")

    paths = {}
    for name, t in [("tidy_dialogs", dia), ("tidy_utterances", utt), ("tidy_entities", ent)]:
        p = OUT / f"{name}_zstd3.parquet"
        t0 = time.perf_counter()
        t.to_parquet(p, engine="pyarrow", compression="zstd", compression_level=3, index=False)
        paths[name] = (t, p, time.perf_counter() - t0)
        log(f"{name} 저장 {mb(p):.2f}MB")

    # ---- 검증 ----
    chk = pd.read_parquet(paths["tidy_utterances"][1])
    assert len(chk) == len(df)
    assert not chk.duplicated(["대화셋일련번호", "문장번호"]).any()
    assert (chk["발화"] != "").all()
    back = chk["발화"].to_numpy()
    orig = df[TEXT_COLS].apply(lambda r: next((x for x in r if x), ""), axis=1).to_numpy()
    assert (back == orig).all(), "발화 값 불일치"
    ai = chk["의도"].astype(str).to_numpy()
    assert ((cust.where(df["화자"] == "고객", agent)).to_numpy() == ai)[(chk["의도기준"] == "화자일치열").to_numpy()].all(), "의도 불일치"
    assert ((chk["보조의도"].astype(str) != "").sum() == both), "보조의도 개수 불일치"
    log("검증 통과")

    sizes = pd.DataFrame({
        "표": list(paths),
        "한 행의 뜻": ["대화 1건", "발화 1줄", "발화 안의 값 1개"],
        "행수": [len(v[0]) for v in paths.values()],
        "열": [", ".join(v[0].columns) for v in paths.values()],
        "MB": [round(mb(v[1]), 2) for v in paths.values()],
        "저장초": [round(v[2], 2) for v in paths.values()],
    })
    cat_before = df["카테고리"].nunique()
    src_cat = pd.read_parquet(BASE, columns=["카테고리"])["카테고리"].astype(str).nunique()
    md = f"""# tidy(long form) 변환 보고서

- 입력: `{BASE.name}` ({len(df):,}행, 대화 {df['대화셋일련번호'].nunique():,}개)
- 원문은 이 보고서에 옮기지 않았어요. 열 이름과 숫자만 적었어요.
- 저장은 중간 산출물용 zstd-3이에요. 원본 base 파일(zstd-19)은 건드리지 않아요.

## 만든 표

{sizes.to_markdown(index=False)}

## 바꾼 점

| 문제 | 처리 |
|---|---|
| 발화문이 4개 열에 나뉨 | `발화` 열 하나로 합침. 모든 행에 값이 정확히 1개라 손실 없음 |
| 의도가 2개 열에 나뉨 | 화자와 같은 쪽 열을 `의도`로 씀(고객 발화는 고객의도, 상담사 발화는 상담사의도). 두 열이 동시에 찬 {int(both)}행은 버려지는 쪽 값을 `보조의도`에 보존. `의도기준`: 화자일치열 / 반대열 / 없음 |
| 카테고리 값 {src_cat}개 | 공백 제거·대소문자 정리 후 {cat_before}개 |
| 도메인 열 | 값이 1개(K쇼핑)뿐이라 뺌 |
| 한 칸에 여러 값(개체명 등) | 쉼표로 쪼개 `tidy_entities`에 한 값씩 저장. 지식베이스·용어사전은 "값/유형"을 `값`·`유형`으로 나눔 |
| 대화 단위 정보 | `tidy_dialogs`로 분리. 카테고리, 총턴수, 발화수, 고객·상담사 발화수 |

## 검증

- 발화 표 {len(chk):,}행, `대화셋일련번호 + 문장번호` 중복 0건
- 4개 원문 열을 합친 값과 `발화` 열이 전부 같음
- 개체 표 행수: {len(ent):,} (출처별 {ent['출처'].value_counts().to_dict()})

## 쓰는 법

```python
import pandas as pd
utt = pd.read_parquet("./data/tidy_utterances_zstd3.parquet", columns=["대화셋일련번호", "문장번호", "화자", "의도"])
dia = pd.read_parquet("./data/tidy_dialogs_zstd3.parquet")
dia.groupby("카테고리")["총턴수"].describe()
```
"""
    md += stats_section(dia, utt, ent)
    (REPORT / "03_tidy_변환_보고서.md").write_text(md, encoding="utf-8")
    update_report2(sizes, len(ent))
    sizes.to_csv(REPORT / "tidy_표_요약.csv", encoding="utf-8-sig", index=False)
    print(sizes.drop(columns=["열"]).to_string(index=False))
    print("의도 동시 존재:", int(both), "카테고리:", src_cat, "->", cat_before)


if __name__ == "__main__":
    main()
