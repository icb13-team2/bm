"""쇼핑 라벨링 데이터 다운캐스트 + parquet 저장 + 마크다운 보고서 생성.

실행 위치: 데이터_다운캐스트/ 폴더
    uv run --no-project --with pandas --with pyarrow python src/downcast_parquet.py
"""
import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd

ZIP_PATH = Path("../쇼핑_라벨링데이터.zip")
DATA = Path("./data")
REPORT = Path("./report")

# 원문 문장이 들어 있는 컬럼: 보고서에는 글자수만 표시
TEXT_COLS = ["고객질문(요청)", "상담사질문(요청)", "고객답변", "상담사답변",
             "개체명", "용어사전", "지식베이스"]


def same(a, b):
    """다시 읽은 값이 원본과 같은지 컬럼별로 문자열 비교."""
    return all((a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all()
               for c in b.columns)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def mb(n):
    return n / 1024 ** 2


def load_raw():
    frames = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".json"))
        raw_json_bytes = sum(zf.getinfo(n).file_size for n in names)
        for n in names:
            with zf.open(n) as f:
                frames.append(pd.DataFrame(json.load(f), dtype=object))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]  # '개체명 ' 뒤 공백 제거
    return df.astype(object), names, raw_json_bytes


def mask(df):
    """미리보기·통계용: 원문 컬럼은 글자수로 바꿔서 보여줌."""
    out = df.copy()
    for c in TEXT_COLS:
        if c in out:
            out[c] = out[c].astype(str).map(lambda s: f"({len(s)}자)" if s else "")
    return out


def info_text(df):
    buf = io.StringIO()
    df.info(buf=buf, memory_usage="deep")
    return buf.getvalue()


def describe_cat(df):
    cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    d = df[cols].astype(str).describe().T
    d["top"] = [f"({len(t)}자)" if (c in TEXT_COLS and t) else (t if t else "(빈값)")
                for c, t in zip(d.index, d["top"])]
    d["빈값비율"] = [(df[c].astype(str) == "").mean().round(3) for c in d.index]
    return d


def save(df, path, codec, level=None):
    t = time.perf_counter()
    df.to_parquet(path, engine="pyarrow", compression=codec,
                  compression_level=level, index=False)
    w = time.perf_counter() - t
    t = time.perf_counter()
    back = pd.read_parquet(path)
    r = time.perf_counter() - t
    return path.stat().st_size, w, r, back


def md_table(df):
    return df.to_markdown()


def main():
    DATA.mkdir(exist_ok=True)
    REPORT.mkdir(exist_ok=True)

    # ---------- 0. 원본 (모든 컬럼 object) ----------
    raw, names, raw_json_bytes = load_raw()
    zip_bytes = ZIP_PATH.stat().st_size
    log("원본 로드 완료")
    mem0 = raw.memory_usage(deep=True).sum()
    p0 = DATA / "shopping_00_raw_object_snappy.parquet"
    size0, _, _, _ = save(raw, p0, "snappy")

    # ---------- 1. downcast ----------
    dc = raw.copy()
    dc["문장번호"] = pd.to_numeric(dc["문장번호"], downcast="unsigned")
    for c in dc.columns:
        if c != "문장번호":
            dc[c] = dc[c].astype("string[pyarrow]")
    mem1 = dc.memory_usage(deep=True).sum()
    p1 = DATA / "shopping_01_downcast_snappy.parquet"
    size1, w1, r1, back1 = save(dc, p1, "snappy")
    assert same(back1, raw), "1단계 값 불일치"
    log("1단계 완료")

    # ---------- 2. category + 압축 방식 비교 ----------
    ct = dc.copy()
    cat_cols, col_mem = [], []
    for c in ct.columns:
        if c == "문장번호":
            continue
        before = ct[c].memory_usage(deep=True)
        as_cat = ct[c].astype("category")
        after = as_cat.memory_usage(deep=True)
        ratio = ct[c].nunique() / len(ct)
        ok = ratio < 0.5 and after < before
        col_mem.append((c, ct[c].nunique(), round(ratio, 4), mb(before), mb(after), ok))
        if ok:
            ct[c] = as_cat
            cat_cols.append(c)
    mem2 = ct.memory_usage(deep=True).sum()
    log("category 변환 완료")

    codecs = [("snappy", None), ("lz4", None), ("gzip", 9), ("brotli", 11),
              ("zstd", 3), ("zstd", 9), ("zstd", 19)]
    rows = []
    for codec, lvl in codecs:
        tag = f"{codec}{'' if lvl is None else '-' + str(lvl)}"
        p = DATA / f"_tmp_{tag}.parquet"
        s, w, r, back = save(ct, p, codec, lvl)
        assert same(back, raw), f"{tag} 값 불일치"
        log(f"{tag} 저장 완료")
        rows.append((tag, s, w, r))
        p.unlink()
    comp = pd.DataFrame(rows, columns=["압축", "bytes", "저장초", "읽기초"])
    # 최종: zstd-19 (brotli-11과 크기는 비슷하고 저장이 훨씬 빨라서 팀 결정으로 선택)
    best = comp.set_index("압축").loc["zstd-19"].copy()
    best["압축"] = "zstd-19"
    bc, bl = (best["압축"].split("-") + [None])[:2]
    p2 = DATA / f"shopping_02_category_{best['압축']}.parquet"
    size2, w2, r2, back2 = save(ct, p2, bc, int(bl) if bl else None)
    p2s = DATA / "shopping_02_category_snappy.parquet"
    size2s, _, _, _ = save(ct, p2s, "snappy")

    # ---------- 보고서 공통 조각 ----------
    head = mask(dc.head(5)).to_markdown(index=False)
    tail = mask(dc.tail(5)).to_markdown(index=False)
    num_desc = dc[["문장번호"]].describe().T.round(2).to_markdown()
    cat_desc1 = describe_cat(dc).to_markdown()
    per_file = raw.groupby("카테고리", observed=True).agg(
        행수=("문장번호", "size"), 대화수=("대화셋일련번호", "nunique"))

    size_tbl1 = pd.DataFrame({
        "단계": ["원본 JSON 7개(압축 해제)", "원본 zip", "메모리: 원본(object)", "메모리: downcast 후",
               "parquet: 원본 object(snappy)", "parquet: downcast 후(snappy)"],
        "MB": [mb(raw_json_bytes), mb(zip_bytes), mb(mem0), mb(mem1), mb(size0), mb(size1)],
    })
    size_tbl1["원본 JSON 대비"] = (size_tbl1["MB"] / mb(raw_json_bytes) * 100).round(1).astype(str) + "%"
    size_tbl1["MB"] = size_tbl1["MB"].round(2)

    common_head = f"""- 원본: `../쇼핑_라벨링데이터.zip` 안의 JSON {len(names)}개 ({', '.join(Path(n).stem for n in names)})
- 합친 크기: {len(raw):,}행 × {raw.shape[1]}열, 대화 {raw['대화셋일련번호'].nunique():,}개
- 재현: `데이터_다운캐스트/` 폴더에서 `uv run --no-project --with pandas --with pyarrow --with tabulate python src/downcast_parquet.py`
- 원문 보호: 미리보기와 통계에서 원문 컬럼({', '.join(TEXT_COLS)})은 글자수 `(N자)`로만 표시했어요.
"""

    # ---------- 보고서 1 ----------
    r1md = f"""# 쇼핑 라벨링 데이터 downcast 보고서 (1차)

{common_head}
## 1. 크기 비교 (downcast 전 vs 후)

{size_tbl1.to_markdown(index=False)}

- 메모리: {mb(mem0):.1f}MB → {mb(mem1):.1f}MB ({(1 - mem1 / mem0) * 100:.1f}% 감소)
- parquet 파일: {mb(size0):.1f}MB → {mb(size1):.1f}MB ({(1 - size1 / size0) * 100:.1f}% 감소)
  - parquet은 원래 열 단위 압축·인코딩을 해서, 타입만 바꿔서는 파일 크기가 거의 그대로예요. downcast 효과는 주로 메모리에서 나와요.
- 값 검증: 다시 읽은 parquet와 원본을 문자열로 비교해 전부 일치했어요.

## 2. downcast 방법

| 컬럼 | 원래 타입 | 바꾼 타입 | 방법 |
|---|---|---|---|
| 문장번호 | object(문자열 "1"~"298") | `{dc['문장번호'].dtype}` | `pd.to_numeric(downcast="unsigned")`. 최대값 {int(dc['문장번호'].max())}이라 2바이트면 충분해요. |
| 나머지 14개 | object(파이썬 문자열) | `string[pyarrow]` | 파이썬 객체 대신 Arrow 문자열 버퍼에 담아 메모리를 줄여요. |
| (공통) | `개체명 ` | `개체명` | 컬럼 이름 끝 공백만 지웠어요. |

- 원래 JSON에는 숫자 타입이 하나도 없어요. 그래서 숫자로 바꿀 수 있는 컬럼은 `문장번호` 하나뿐이에요.
- 빈 문자열(`""`)은 그대로 뒀어요. "해당 없음" 뜻이라 NaN으로 바꾸면 의미가 바뀔 수 있어서요.
- 저장: `to_parquet(engine="pyarrow", compression="snappy")` (pandas 기본값)

## 3. 미리보기

### 상단 5행
{head}

### 하단 5행
{tail}

## 4. info (downcast 후)

```text
{info_text(dc)}```

## 5. 기술 통계

### 수치형
{num_desc}

### 범주형(문자열)
{cat_desc1}

### 카테고리별 행수·대화수
{per_file.to_markdown()}
"""
    (REPORT / "01_downcast_보고서.md").write_text(r1md, encoding="utf-8")

    # ---------- 보고서 2 ----------
    colmem = pd.DataFrame(col_mem, columns=["컬럼", "고유값수", "고유값비율", "string MB", "category MB", "category 적용"])
    colmem[["string MB", "category MB"]] = colmem[["string MB", "category MB"]].round(2)
    comp_show = comp.copy()
    comp_show["MB"] = (comp_show["bytes"] / 1024 ** 2).round(2)
    comp_show["저장초"] = comp_show["저장초"].round(2)
    comp_show["읽기초"] = comp_show["읽기초"].round(2)
    comp_show = comp_show[["압축", "MB", "저장초", "읽기초"]].sort_values("MB")

    size_tbl2 = pd.DataFrame({
        "단계": ["원본 JSON 7개(압축 해제)", "원본 zip", "parquet: 원본 object(snappy)",
               "parquet: downcast(snappy)", "parquet: category(snappy)", f"parquet: category({best['압축']}) ← 최종"],
        "MB": [mb(raw_json_bytes), mb(zip_bytes), mb(size0), mb(size1), mb(size2s), mb(size2)],
    })
    size_tbl2["원본 JSON 대비"] = (size_tbl2["MB"] / mb(raw_json_bytes) * 100).round(2).astype(str) + "%"
    size_tbl2["MB"] = size_tbl2["MB"].round(2)
    mem_tbl = pd.DataFrame({"단계": ["원본(object)", "downcast", "downcast+category"],
                            "메모리 MB": [round(mb(m), 1) for m in (mem0, mem1, mem2)]})

    r2md = f"""# 쇼핑 라벨링 데이터 category + 고압축 보고서 (2차)

{common_head}
## 1. 크기 비교

### 파일 크기
{size_tbl2.to_markdown(index=False)}

### 메모리
{mem_tbl.to_markdown(index=False)}

- 최종 파일: `data/{p2.name}` ({mb(size2):.2f}MB)
- 1차(downcast, snappy) 대비 {(1 - size2 / size1) * 100:.1f}% 더 작아졌어요.
- 메모리는 원본 대비 {(1 - mem2 / mem0) * 100:.1f}% 줄었어요.
- 값 검증: 모든 압축 방식에서 다시 읽은 값이 원본과 전부 일치했어요.

## 2. category 변환 방법

- 기준: `고유값수 / 전체행수 < 0.5` 이고, category로 바꿨을 때 메모리가 실제로 줄어드는 컬럼.
- `문장번호`는 이미 `uint16` 숫자라 제외했어요. 순서·크기 비교가 필요한 값이라서요.
- 결과: {len(cat_cols)}개 컬럼을 category로 바꿨어요.

{colmem.to_markdown(index=False)}

- 대화 데이터는 같은 문장·라벨이 많이 반복돼요. 빈 문자열도 많고요. 그래서 원문 컬럼까지 category 효과가 있어요.

## 3. 압축 방식 비교 (category 적용 데이터)

{comp_show.to_markdown(index=False)}

- 최종 저장은 `{best['압축']}`이에요. brotli-11이 조금 더 작지만 저장이 훨씬 느려서, 자주 다시 저장하기 편한 쪽으로 골랐어요.
- parquet는 category를 dictionary 인코딩으로 저장해요. 다시 읽어도 category 타입이 그대로 살아나요.

## 4. 미리보기

### 상단 5행
{mask(ct.head(5)).to_markdown(index=False)}

### 하단 5행
{mask(ct.tail(5)).to_markdown(index=False)}

## 5. info (category 적용 후)

```text
{info_text(ct)}```

## 6. 기술 통계

### 수치형
{ct[["문장번호"]].describe().T.round(2).to_markdown()}

### 범주형
{describe_cat(ct).to_markdown()}

## 7. 분석에 더 좋을까?

- 네, 대체로 좋아요. 읽기가 훨씬 빠르고 가벼워져요. JSON 7개 파싱 대신 parquet 한 파일만 읽으면 돼요.
- 라벨 컬럼(카테고리, 화자, QA, 고객의도, 상담사의도)은 category가 groupby·value_counts에 특히 유리해요.
- 컬럼만 골라 읽을 수 있어요. 예: `pd.read_parquet(path, columns=["대화셋일련번호", "문장번호", "고객의도"])`
- 주의 1: 원문 컬럼에 `.str` 함수를 쓰려면 먼저 `.astype("string")`으로 바꿔 주세요. category는 문자열 처리용이 아니에요.
- 주의 2: 이 데이터의 핵심 분석(턴수·흐름·BERT)은 결국 원문 텍스트를 써요. 그래서 이득은 "속도·용량" 쪽이지, 분석 결과 자체가 바뀌진 않아요.
- 주의 3: zstd-19 같은 높은 압축은 저장이 느려요. 자주 다시 저장하는 중간 산출물은 snappy나 zstd-3이 편해요.
"""
    (REPORT / "02_category_압축_보고서.md").write_text(r2md, encoding="utf-8")

    summary = pd.DataFrame({
        "파일": [p0.name, p1.name, p2s.name, p2.name],
        "MB": [round(mb(s), 2) for s in (size0, size1, size2s, size2)],
    })
    summary.to_csv(REPORT / "크기비교_요약.csv", encoding="utf-8-sig", index=False)
    print(size_tbl2.to_string(index=False))
    print(mem_tbl.to_string(index=False))
    print(comp_show.to_string(index=False))
    print("category:", cat_cols)


if __name__ == "__main__":
    main()
