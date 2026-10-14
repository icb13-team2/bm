"""표본 60건 독립 검수용 대시보드 HTML 생성.

표본ID를 고르면 판정 대상 고객 발화, 직전 2발화, 직후 상담사 응답,
대화 내 위치를 보여 준다. 대화 전체 펼쳐 보기도 된다.
기존 고객의도, 규칙 탐지, JEV 결과, 표본 설계 열은 넣지 않는다.

실행 예시 (프로젝트 루트에서):
  uv run --no-project --with pandas --with pyarrow --with openpyxl \
    python src/build_review_dashboard.py --design <표본설계.xlsx>
"""

import argparse
import json
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UTT_PATH = ROOT / "data" / "tidy_utterances_zstd3.parquet"
OUT_PATH = ROOT / "report" / "04_표본검수_대시보드.html"
DESIGN_SHEET = "99_관리자_표본설계"


def load_samples(design_path: Path) -> list[dict]:
    """표본 설계 시트에서 표본ID, 대화ID, 턴번호, 카테고리만 읽는다."""
    wb = openpyxl.load_workbook(design_path, read_only=True)
    rows = list(wb[DESIGN_SHEET].iter_rows(values_only=True))
    samples = []
    for r in rows[1:]:
        if not r[0]:
            continue
        samples.append(
            {
                "sid": str(r[0]),
                "dialog": str(r[1]),
                "turn": int(r[2]),
                "cat": str(r[3]),
            }
        )
    return samples


def build_payload(samples: list[dict]) -> dict:
    utt = pd.read_parquet(UTT_PATH, columns=["대화셋일련번호", "문장번호", "화자", "발화"])
    utt["대화셋일련번호"] = utt["대화셋일련번호"].astype(str)
    utt["화자"] = utt["화자"].astype(str)
    dialogs = set(s["dialog"] for s in samples)
    utt = utt[utt["대화셋일련번호"].isin(dialogs)].sort_values(["대화셋일련번호", "문장번호"])

    convs = {}
    for dialog, grp in utt.groupby("대화셋일련번호"):
        convs[dialog] = [
            {"n": int(n), "s": str(sp), "t": str(tx)}
            for n, sp, tx in zip(grp["문장번호"], grp["화자"], grp["발화"])
        ]

    items = []
    for s in samples:
        conv = convs[s["dialog"]]
        idx = next(i for i, u in enumerate(conv) if u["n"] == s["turn"])
        prev = conv[max(0, idx - 2) : idx]
        nxt = next((u for u in conv[idx + 1 :] if u["s"] == "상담사"), None)
        items.append(
            {
                "sid": s["sid"],
                "dialog": s["dialog"],
                "cat": s["cat"],
                "turn": s["turn"],
                "total": len(conv),
                "pos": idx + 1,
                "prev": prev,
                "target": conv[idx],
                "agent": nxt,
                "conv": conv,
            }
        )
    return {"items": items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", required=True, type=Path, help="표본 설계가 든 검수 엑셀")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    samples = load_samples(args.design)
    payload = build_payload(samples)

    # 스크립트 안의 </ 문자열이 HTML을 깨지 않도록 이스케이프한다
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA__", data_json)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"표본 {len(payload['items'])}건, 발화 {sum(len(i['conv']) for i in payload['items'])}행 기록")
    print(f"저장: {args.out}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>표본 검수 대시보드 (60건)</title>
<style>
:root {
  --bg: #f4f6fa; --card: #ffffff; --line: #e2e8f0; --text: #0f172a;
  --muted: #64748b; --primary: #2563eb; --primary-soft: #dbeafe;
  --target: #fef3c7; --target-line: #f59e0b; --agent: #e0f2fe;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0b101b; --card: #131a29; --line: #242f46; --text: #e2e8f0;
    --muted: #94a3b8; --primary: #60a5fa; --primary-soft: #1e3a5f;
    --target: #3b2f10; --target-line: #f59e0b; --agent: #14314a;
  }
}
:root[data-theme="dark"] {
  --bg: #0b101b; --card: #131a29; --line: #242f46; --text: #e2e8f0;
  --muted: #94a3b8; --primary: #60a5fa; --primary-soft: #1e3a5f;
  --target: #3b2f10; --target-line: #f59e0b; --agent: #14314a;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: "Pretendard", "Malgun Gothic", -apple-system, "Segoe UI", sans-serif;
  line-height: 1.6;
}
.wrap { max-width: 880px; margin: 0 auto; padding: 16px; }
header h1 { font-size: 20px; margin: 8px 0 4px; }
header p { margin: 0; color: var(--muted); font-size: 13px; }
.notice {
  margin: 12px 0; padding: 10px 12px; border-radius: 8px;
  background: var(--card); border: 1px solid var(--line); font-size: 13px; color: var(--muted);
}
.toolbar {
  display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
  background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px;
}
select, button {
  font: inherit; font-size: 14px; color: var(--text); background: var(--card);
  border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; cursor: pointer;
}
select { flex: 1 1 220px; min-width: 0; }
button.primary { background: var(--primary); color: #fff; border-color: var(--primary); }
button:disabled { opacity: 0.4; cursor: default; }
.card {
  margin-top: 12px; background: var(--card); border: 1px solid var(--line);
  border-radius: 12px; padding: 16px;
}
.meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.chip {
  font-size: 12px; padding: 2px 8px; border-radius: 999px;
  background: var(--primary-soft); color: var(--primary);
}
.section-title { font-size: 13px; font-weight: 700; color: var(--muted); margin: 16px 0 6px; }
.bubble {
  padding: 10px 12px; border-radius: 10px; margin: 6px 0;
  border: 1px solid var(--line); white-space: pre-wrap; word-break: keep-all;
}
.bubble .who { display: block; font-size: 12px; color: var(--muted); margin-bottom: 2px; }
.bubble.prev { background: var(--bg); }
.bubble.target { background: var(--target); border: 2px solid var(--target-line); }
.bubble.agent { background: var(--agent); }
.empty { color: var(--muted); font-size: 14px; padding: 8px 0; }
.full-toggle { margin-top: 16px; width: 100%; }
.full { margin-top: 10px; max-height: 520px; overflow-y: auto; border-top: 1px solid var(--line); padding-top: 10px; }
.full .bubble { font-size: 14px; }
.foot { margin-top: 14px; font-size: 12px; color: var(--muted); }
@media (max-width: 480px) {
  .wrap { padding: 12px; }
  .bubble { font-size: 15px; }
}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>표본 검수 대시보드 (60건)</h1>
    <p>표본ID를 고르면 판정 대상 발화를 보여 줍니다. 팀원 판정과 다른 시트는 보이지 않습니다.</p>
  </header>

  <div class="notice">
    이 파일에는 고객 원문이 들어 있습니다. 검수 목적 외에는 공유하지 마세요.
  </div>

  <div class="toolbar">
    <button id="prev" type="button">이전</button>
    <select id="sid-select" aria-label="표본ID 선택"></select>
    <button id="next" type="button">다음</button>
    <select id="cat-select" aria-label="카테고리 필터"></select>
  </div>

  <div class="card" id="card"></div>
  <div class="foot" id="foot"></div>
</div>

<script>
const DATA = __DATA__;
const ITEMS = DATA.items;
let filtered = ITEMS.slice();
let current = 0;

const $ = (id) => document.getElementById(id);

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

function bubble(u, cls, label) {
  return `<div class="bubble ${cls}"><span class="who">${esc(label)} · ${u.n}턴</span>${esc(u.t)}</div>`;
}

function initFilters() {
  const cats = [...new Set(ITEMS.map((i) => i.cat))];
  $("cat-select").innerHTML =
    `<option value="">전체 카테고리</option>` +
    cats.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
  $("sid-select").innerHTML = ITEMS.map(
    (i) => `<option value="${esc(i.sid)}">${esc(i.sid)} · ${esc(i.cat)}</option>`
  ).join("");
}

function applyFilter() {
  const cat = $("cat-select").value;
  filtered = ITEMS.filter((i) => !cat || i.cat === cat);
  $("sid-select").innerHTML = filtered.map(
    (i) => `<option value="${esc(i.sid)}">${esc(i.sid)} · ${esc(i.cat)}</option>`
  ).join("");
  current = 0;
  render();
}

function render() {
  if (!filtered.length) {
    $("card").innerHTML = `<div class="empty">표본이 없습니다.</div>`;
    return;
  }
  const item = filtered[current];
  $("sid-select").value = item.sid;
  $("prev").disabled = current === 0;
  $("next").disabled = current === filtered.length - 1;

  const prevHtml = item.prev.length
    ? item.prev.map((u) => bubble(u, "prev", u.s)).join("")
    : `<div class="empty">앞 발화가 없습니다. 대화 첫머리입니다.</div>`;
  const agentHtml = item.agent
    ? bubble(item.agent, "agent", "상담사")
    : `<div class="empty">직후 상담사 응답이 없습니다.</div>`;
  const t = item.target;

  $("card").innerHTML = `
    <div class="meta">
      <span class="chip">표본 ${esc(item.sid)}</span>
      <span class="chip">${esc(item.cat)}</span>
      <span class="chip">${item.turn}턴 · 대화 ${item.pos}/${item.total}턴</span>
    </div>
    <div class="section-title">직전 문맥 (최대 2발화)</div>
    ${prevHtml}
    <div class="section-title">판정 대상 고객 발화</div>
    ${bubble(t, "target", "고객")}
    <div class="section-title">직후 상담사 응답</div>
    ${agentHtml}
    <button class="full-toggle" id="full-btn" type="button">대화 전체 펼쳐 보기</button>
    <div class="full" id="full" hidden>
      ${item.conv.map((u) => bubble(u, u.n === t.n ? "target" : (u.s === "고객" ? "" : "agent"), u.s)).join("")}
    </div>`;

  $("full-btn").onclick = () => {
    const box = $("full");
    box.hidden = !box.hidden;
    $("full-btn").textContent = box.hidden ? "대화 전체 펼쳐 보기" : "대화 전체 접기";
    if (!box.hidden) {
      const target = box.querySelector(".bubble.target");
      if (target) target.scrollIntoView({ block: "center" });
    }
  };
  $("foot").textContent = `필터 적용 ${filtered.length}건 중 ${current + 1}번째`;
}

$("prev").onclick = () => { if (current > 0) { current--; render(); } };
$("next").onclick = () => { if (current < filtered.length - 1) { current++; render(); } };
$("sid-select").onchange = (e) => {
  const idx = filtered.findIndex((i) => i.sid === e.target.value);
  if (idx >= 0) { current = idx; render(); }
};
$("cat-select").onchange = applyFilter;

initFilters();
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
