"""전체 상담 스크립트 조회용 뷰어 생성 (index.html + data/*.js).

대화 26,270개의 원문을 묶음 파일(chunk)로 나눠 저장한다.
index.html은 더블클릭으로 열리고 필요한 묶음만 <script>로 불러온다.
(file:// 에서는 fetch가 막히므로 .js 파일을 script 태그로 읽는 방식을 쓴다.)

라벨, 의도, 규칙 탐지, JEV 결과는 넣지 않는다. 화자와 발화 순서만 넣는다.
원문이 들어가므로 출력 폴더는 GitHub에 올리지 않는다.

실행 예시 (프로젝트 루트에서):
  uv run --no-project --with pandas --with pyarrow \
    python src/build_full_viewer.py --out <출력폴더>
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UTT_PATH = ROOT / "data" / "tidy_utterances_zstd3.parquet"
DLG_PATH = ROOT / "data" / "tidy_dialogs_zstd3.parquet"
CHUNK_SIZE = 1000
PREVIEW_LEN = 28


def num_key(dialog_id: str) -> tuple:
    m = re.search(r"\d+", dialog_id)
    return (int(m.group()) if m else 0, dialog_id)


def js_dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out: Path = args.out
    (out / "data").mkdir(parents=True, exist_ok=True)

    dlg = pd.read_parquet(DLG_PATH)
    dlg["대화셋일련번호"] = dlg["대화셋일련번호"].astype(str)
    dlg["카테고리"] = dlg["카테고리"].astype(str)
    utt = pd.read_parquet(UTT_PATH, columns=["대화셋일련번호", "문장번호", "화자", "발화"])
    utt["대화셋일련번호"] = utt["대화셋일련번호"].astype(str)
    utt["화자"] = utt["화자"].astype(str)
    utt["발화"] = utt["발화"].astype(str)
    utt = utt.sort_values(["대화셋일련번호", "문장번호"])

    ids = sorted(dlg["대화셋일련번호"].tolist(), key=num_key)
    cat_of = dict(zip(dlg["대화셋일련번호"], dlg["카테고리"]))
    cats = sorted(set(cat_of.values()))
    cat_idx = {c: i for i, c in enumerate(cats)}
    chunk_of = {d: i // CHUNK_SIZE for i, d in enumerate(ids)}

    groups = {d: g for d, g in utt.groupby("대화셋일련번호")}
    n_chunks = (len(ids) + CHUNK_SIZE - 1) // CHUNK_SIZE

    index_rows = []
    total_bytes = 0
    for c in range(n_chunks):
        rows = []
        for d in ids[c * CHUNK_SIZE : (c + 1) * CHUNK_SIZE]:
            g = groups[d]
            turns = [
                [int(n), 0 if sp == "고객" else 1, tx]
                for n, sp, tx in zip(g["문장번호"], g["화자"], g["발화"])
            ]
            rows.append([d, turns])
            first_cust = next((t[2] for t in turns if t[1] == 0), "")
            index_rows.append([d, cat_idx[cat_of[d]], len(turns), c, first_cust[:PREVIEW_LEN]])
        text = f"window.__chunk({c},{js_dump(rows)});\n"
        (out / "data" / f"chunk_{c:03d}.js").write_text(text, encoding="utf-8")
        total_bytes += len(text.encode("utf-8"))

    idx_text = f"window.__index({js_dump({'cats': cats, 'rows': index_rows, 'chunks': n_chunks})});\n"
    (out / "data" / "index_data.js").write_text(idx_text, encoding="utf-8")
    (out / "index.html").write_text(HTML, encoding="utf-8")

    print(f"대화 {len(ids):,}개, 발화 {len(utt):,}행, 묶음 {n_chunks}개")
    print(f"묶음 파일 합계 {total_bytes/1e6:.1f}MB, 평균 {total_bytes/n_chunks/1e6:.2f}MB")
    print(f"index_data.js {len(idx_text.encode('utf-8'))/1e6:.2f}MB")


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>전체 상담 스크립트 뷰어</title>
<style>
:root {
  --bg:#f4f6fa; --card:#fff; --line:#e2e8f0; --text:#0f172a; --muted:#64748b;
  --primary:#2563eb; --soft:#dbeafe; --cust:#f1f5f9; --agent:#e0f2fe; --hit:#fde047;
}
@media (prefers-color-scheme: dark) {
  :root { --bg:#0b101b; --card:#131a29; --line:#242f46; --text:#e2e8f0; --muted:#94a3b8;
    --primary:#60a5fa; --soft:#1e3a5f; --cust:#1c2538; --agent:#14314a; --hit:#a16207; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); line-height:1.6;
  font-family:"Pretendard","Malgun Gothic",-apple-system,"Segoe UI",sans-serif; }
.app { display:grid; grid-template-columns:360px 1fr; height:100vh; }
.side { border-right:1px solid var(--line); background:var(--card); display:flex; flex-direction:column; min-height:0; }
.side-head { padding:12px; border-bottom:1px solid var(--line); }
.side-head h1 { font-size:16px; margin:0 0 8px; }
input[type=search], select { width:100%; font:inherit; font-size:14px; color:var(--text);
  background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 10px; margin-bottom:6px; }
.row { display:flex; gap:6px; }
.row > * { flex:1; min-width:0; }
button { font:inherit; font-size:13px; color:var(--text); background:var(--card);
  border:1px solid var(--line); border-radius:8px; padding:7px 10px; cursor:pointer; }
button.primary { background:var(--primary); border-color:var(--primary); color:#fff; }
.status { font-size:12px; color:var(--muted); padding:6px 12px; border-bottom:1px solid var(--line); }
.bar { height:4px; background:var(--line); } .bar > div { height:4px; width:0; background:var(--primary); }
.list { flex:1; overflow-y:auto; min-height:0; }
.item { padding:8px 12px; border-bottom:1px solid var(--line); cursor:pointer; }
.item:hover { background:var(--cust); } .item.active { background:var(--soft); }
.item .top { display:flex; justify-content:space-between; font-size:13px; font-weight:600; }
.item .sub { font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.main { overflow-y:auto; padding:16px; min-height:0; }
.title { font-size:16px; font-weight:700; margin-bottom:4px; }
.chips { display:flex; gap:6px; margin-bottom:12px; flex-wrap:wrap; }
.chip { font-size:12px; padding:2px 8px; border-radius:999px; background:var(--soft); color:var(--primary); }
.msg { max-width:78%; margin:6px 0; padding:8px 12px; border-radius:12px; white-space:pre-wrap; word-break:keep-all; }
.msg .who { display:block; font-size:11px; color:var(--muted); }
.msg.c { background:var(--cust); border:1px solid var(--line); }
.msg.a { background:var(--agent); margin-left:auto; }
mark { background:var(--hit); color:inherit; padding:0 1px; border-radius:2px; }
.empty { color:var(--muted); padding:24px 12px; }
.back { display:none; }
@media (max-width: 760px) {
  .app { grid-template-columns:1fr; }
  .main { display:none; } .app.viewing .side { display:none; } .app.viewing .main { display:block; }
  .back { display:inline-block; margin-bottom:8px; } .msg { max-width:92%; }
}
</style>
</head>
<body>
<div class="app" id="app">
  <aside class="side">
    <div class="side-head">
      <h1>전체 상담 스크립트</h1>
      <input type="search" id="q-id" placeholder="대화ID (예: S22987)">
      <select id="q-cat"></select>
      <input type="search" id="q-kw" placeholder="키워드 (원문 전체 검색)">
      <div class="row">
        <button class="primary" id="kw-go" type="button">키워드 검색</button>
        <button id="kw-stop" type="button" disabled>중지</button>
      </div>
    </div>
    <div class="bar"><div id="bar"></div></div>
    <div class="status" id="status"></div>
    <div class="list" id="list"></div>
  </aside>
  <section class="main" id="main"><div class="empty">왼쪽에서 대화를 고르세요.</div></section>
</div>

<script>
let INDEX = null;                 // {cats, rows:[id,cat,turns,chunk,preview], chunks}
const waiting = {};               // chunk no -> resolver
const cache = new Map();          // 최근 본 묶음 (최대 3개)
let kwHits = null;                // Map id -> 일치 발화 수 (키워드 검색 결과)
let kwWord = "";
let scanToken = 0;
let shown = 0;
let filtered = [];
let activeId = null;
const PAGE = 200;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

window.__index = (d) => { INDEX = d; init(); };
window.__chunk = (n, rows) => { const r = waiting[n]; if (r) { delete waiting[n]; r(rows); } };

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = src; s.onload = () => { s.remove(); res(); };
    s.onerror = () => { s.remove(); rej(new Error(src)); };
    document.head.appendChild(s);
  });
}
function getChunk(n, keep) {
  if (cache.has(n)) return Promise.resolve(cache.get(n));
  return new Promise((res, rej) => {
    waiting[n] = (rows) => {
      if (keep) { cache.set(n, rows); if (cache.size > 3) cache.delete(cache.keys().next().value); }
      res(rows);
    };
    loadScript(`data/chunk_${String(n).padStart(3, "0")}.js`).catch((e) => { delete waiting[n]; rej(e); });
  });
}

function init() {
  $("q-cat").innerHTML = `<option value="">전체 카테고리</option>` +
    INDEX.cats.map((c, i) => `<option value="${i}">${esc(c)}</option>`).join("");
  $("q-id").oninput = applyFilter;
  $("q-cat").onchange = applyFilter;
  $("kw-go").onclick = runKeyword;
  $("q-kw").onkeydown = (e) => { if (e.key === "Enter") runKeyword(); };
  $("kw-stop").onclick = () => { scanToken++; setBar(0); $("kw-stop").disabled = true; $("status").textContent = "검색을 중지했습니다."; };
  applyFilter();
}

function applyFilter() {
  const q = $("q-id").value.trim().toLowerCase();
  const cat = $("q-cat").value;
  filtered = INDEX.rows.filter((r) =>
    (!q || r[0].toLowerCase().includes(q)) &&
    (cat === "" || String(r[1]) === cat) &&
    (!kwHits || kwHits.has(r[0])));
  shown = 0; $("list").innerHTML = ""; more();
  const kwNote = kwHits ? ` · 키워드 결과 ${kwHits.size.toLocaleString()}개 대화` : "";
  $("status").textContent = `${filtered.length.toLocaleString()}개 대화 (전체 ${INDEX.rows.length.toLocaleString()})${kwNote}`;
}

function more() {
  const part = filtered.slice(shown, shown + PAGE);
  const html = part.map((r) => {
    const hit = kwHits ? ` · 일치 ${kwHits.get(r[0])}` : "";
    return `<div class="item${r[0] === activeId ? " active" : ""}" data-id="${esc(r[0])}">
      <div class="top"><span>${esc(r[0])}</span><span>${esc(INDEX.cats[r[1]])} · ${r[2]}발화${hit}</span></div>
      <div class="sub">${esc(r[4])}</div></div>`;
  }).join("");
  const old = $("list").querySelector(".morebtn"); if (old) old.remove();
  $("list").insertAdjacentHTML("beforeend", html);
  shown += part.length;
  if (shown < filtered.length) {
    $("list").insertAdjacentHTML("beforeend", `<div class="empty morebtn"><button type="button">더 보기 (${(filtered.length - shown).toLocaleString()}개 남음)</button></div>`);
    $("list").querySelector(".morebtn button").onclick = more;
  }
}
$("list").addEventListener("click", (e) => {
  const it = e.target.closest(".item"); if (it) openDialog(it.dataset.id);
});

function setBar(p) { $("bar").style.width = `${Math.round(p * 100)}%`; }

async function runKeyword() {
  const w = $("q-kw").value.trim();
  if (!w) { kwHits = null; kwWord = ""; applyFilter(); return; }
  const token = ++scanToken; kwWord = w;
  const lw = w.toLowerCase(); const hits = new Map();
  $("kw-stop").disabled = false;
  for (let c = 0; c < INDEX.chunks; c++) {
    if (token !== scanToken) return;
    $("status").textContent = `원문 검색 중… ${c + 1}/${INDEX.chunks} 묶음`;
    setBar((c + 1) / INDEX.chunks);
    let rows;
    try { rows = await getChunk(c, false); } catch (e) { $("status").textContent = `묶음 ${c}을 불러오지 못했어요. data 폴더가 index.html 옆에 있는지 확인하세요.`; return; }
    for (const [id, turns] of rows) {
      let n = 0;
      for (const t of turns) if (t[2].toLowerCase().includes(lw)) n++;
      if (n) hits.set(id, n);
    }
  }
  if (token !== scanToken) return;
  kwHits = hits; $("kw-stop").disabled = true; setBar(0); applyFilter();
}

function markWord(text) {
  const s = esc(text);
  if (!kwWord) return s;
  const w = esc(kwWord).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return s.replace(new RegExp(w, "gi"), (m) => `<mark>${m}</mark>`);
}

async function openDialog(id) {
  activeId = id;
  document.querySelectorAll(".item").forEach((el) => el.classList.toggle("active", el.dataset.id === id));
  const r = INDEX.rows.find((x) => x[0] === id);
  $("main").innerHTML = `<div class="empty">불러오는 중…</div>`;
  $("app").classList.add("viewing");
  let rows;
  try { rows = await getChunk(r[3], true); } catch (e) { $("main").innerHTML = `<div class="empty">묶음 파일을 불러오지 못했어요. data 폴더가 index.html 옆에 있는지 확인하세요.</div>`; return; }
  const found = rows.find((x) => x[0] === id);
  const bubbles = found[1].map((t) =>
    `<div class="msg ${t[1] === 0 ? "c" : "a"}"><span class="who">${t[1] === 0 ? "고객" : "상담사"} · ${t[0]}</span>${markWord(t[2])}</div>`).join("");
  $("main").innerHTML = `<button class="back" type="button" id="back">목록으로</button>
    <div class="title">${esc(id)}</div>
    <div class="chips"><span class="chip">${esc(INDEX.cats[r[1]])}</span><span class="chip">${found[1].length}발화</span></div>${bubbles}`;
  $("back").onclick = () => $("app").classList.remove("viewing");
  $("main").scrollTop = 0;
}

loadScript("data/index_data.js").catch(() => {
  $("status").textContent = "data/index_data.js를 찾지 못했어요. index.html과 data 폴더를 같이 두세요.";
});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
