import json, os, math
from typing import Dict, Any, List, Tuple, Optional
import streamlit as st
from streamlit.components.v1 import html as st_html

# ---------- Streamlit Config ----------
st.set_page_config(page_title="Tournament Bracket (Trace)", layout="wide")

# ---------- Helper Functions ----------
def fmt_token(tok: str) -> str:
    """Format tokens for better readability."""
    if not isinstance(tok, str):
        return str(tok)
    t = tok
    if t.startswith("Ġ"):
        t = "·" + t[1:]
    t = t.replace("Ċ", "⏎")
    return t

def load_trace_from_path(path: str) -> Dict[str, Any]:
    """Load JSON trace from a local file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_trace_from_uploader(uploaded) -> Dict[str, Any]:
    """Load JSON trace from uploaded Streamlit file."""
    return json.loads(uploaded.read().decode("utf-8"))

def layout_layers(trace: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Compute visual layout info for tournament layers."""
    layers: List[Dict[str, Any]] = trace.get("layers", [])
    group_size = int(trace.get("group_size", 2))
    for lyr in layers:
        if "groups" not in lyr:
            lyr["groups"] = []
        for g in lyr["groups"]:
            g.setdefault("candidates", [])
            g.setdefault("winner", {})
            g.setdefault("tie", False)
            if len(g["candidates"]) < group_size:
                g["candidates"] += [{}] * (group_size - len(g["candidates"]))
    return layers, [[g for g in lyr["groups"]] for lyr in layers]

# ---------- HTML Builder ----------
def build_html(
    trace: Dict[str, Any],
    card_w: int = 230,
    card_h: int = 56,
    card_gap_y: int = 16,
    col_gap_x: int = 120,
    padding: int = 24
) -> str:
    """Generate full HTML+SVG bracket visualization."""
    if not isinstance(trace, dict):
        raise ValueError("Invalid trace format — expected a dictionary, got None or wrong type.")

    layers = trace.get("layers", [])
    group_size = int(trace.get("group_size", 2))
    step = trace.get("step", "?")
    seed_rt = trace.get("seed_rt", "?")
    m_layers = trace.get("m_layers", "?")
    H = trace.get("H_window", "?")
    nsec = trace.get("nsec", "?")
    g_dist = trace.get("g_dist", "?")

    cols = len(layers) + 2
    col_x = [padding + i * (card_w + col_gap_x) for i in range(cols)]
    init = trace.get("initial_candidates", [])
    init_rows = math.ceil(len(init) / 2)
    max_groups = max([len(lyr.get("groups", [])) for lyr in layers] + [1])
    group_block_h = card_h * group_size + card_gap_y
    canvas_h = padding*2 + max_groups * group_block_h

    # Compute positions
    layer_positions: List[List[Dict[str, Any]]] = []
    for L, lyr in enumerate(layers, start=1):
        groups = lyr.get("groups", [])
        col_idx = 1 + (L - 1)
        x_left = col_x[col_idx]
        total_h = len(groups) * group_block_h
        y_base = padding + (canvas_h - 2*padding - total_h) / 2.0
        pos_groups = []
        for gi, g in enumerate(groups):
            y_top = y_base + gi * group_block_h
            cand_boxes = []
            for ci, cand in enumerate(g.get("candidates", [])):
                box = {
                    "x": x_left,
                    "y": y_top + ci * (card_h),
                    "w": card_w,
                    "h": card_h,
                    "token": fmt_token(cand.get("token", "")),
                    "id": cand.get("id", ""),
                    "g": cand.get("g", None),
                    "winner": cand.get("id", None) == g.get("winner", {}).get("id")
                }
                cand_boxes.append(box)
            win_mid_x = x_left + card_w + (col_gap_x // 2)
            win_mid_y = y_top + (card_h * group_size)/2 - (card_h/2)
            pos_groups.append({
                "cand": cand_boxes,
                "win_mid": (win_mid_x, win_mid_y),
                "winner_id": g.get("winner", {}).get("id"),
                "tie": bool(g.get("tie", False)),
                "layer": L,
                "group_index": gi
            })
        layer_positions.append(pos_groups)

    final_x = col_x[-1]
    final_y = padding + (canvas_h - 2*padding)/2 - (card_h/2)
    svg_lines: List[str] = []

    def box_center_right(b): return (b["x"] + b["w"], b["y"] + b["h"]/2)

    # Draw candidate connections
    for pos_layer in layer_positions:
        for g in pos_layer:
            mx, my = g["win_mid"]
            for b in g["cand"]:
                x2, y2 = box_center_right(b)
                svg_lines.append(f'<path d="M{x2},{y2} H{mx} V{my}" stroke="#667fb2" stroke-width="1.6" fill="none" />')

    # Connect layers
    for L in range(len(layer_positions)-1):
        current = layer_positions[L]
        nxt = layer_positions[L+1]
        for k, g in enumerate(current):
            if k >= len(nxt): break
            mx, my = g["win_mid"]
            target_group = nxt[k]
            tx = target_group["cand"][0]["x"] - (col_gap_x // 2)
            ty = target_group["cand"][0]["y"] + card_h/2
            svg_lines.append(f'<path d="M{mx},{my} H{tx} V{ty}" stroke="#a6b6d6" stroke-width="1.4" fill="none" stroke-dasharray="3,3" />')

    if layer_positions:
        last_layer = layer_positions[-1]
        if len(last_layer) > 0:
            mx, my = last_layer[0]["win_mid"]
            svg_lines.append(f'<path d="M{mx},{my} H{final_x - (col_gap_x//2)} V{final_y + card_h/2}" stroke="#22c55e" stroke-width="2" fill="none" />')

    def card_html(x, y, w, h, title, subtitle, winner=False):
        border = "#22c55e" if winner else "rgba(102,127,178,0.55)"
        bg = "rgba(34,197,94,0.08)" if winner else "rgba(255,255,255,0.03)"
        return f'''
        <div class="card" style="left:{x}px; top:{y}px; width:{w}px; height:{h}px; border-color:{border}; background:{bg};">
          <div class="tok">{title}</div>
          <div class="sub">{subtitle}</div>
        </div>'''

    cards_html: List[str] = []
    init_x = col_x[0]
    init_y_base = padding
    for i, c in enumerate(init):
        y = init_y_base + i * (card_h + 6)
        title = fmt_token(c.get("token", ""))
        sub = f'id #{c.get("id","")}&nbsp;&nbsp; p={c.get("p",0):.4g}'
        cards_html.append(card_html(init_x, y, card_w, card_h, title, sub, winner=False))

    for pos_layer in layer_positions:
        for g in pos_layer:
            for b in g["cand"]:
                t = b["token"] if b["token"] else "—"
                sid = b["id"]
                sub = f'id #{sid}' + (f' · g={b["g"]:.3g}' if b.get("g") is not None else "")
                cards_html.append(card_html(b["x"], b["y"], b["w"], b["h"], t, sub, winner=b["winner"]))

    fin = trace.get("final_winner", {})
    final_title = fmt_token(fin.get("token", "?"))
    final_sub = f'id #{fin.get("id","?")}'
    cards_html.append(card_html(final_x, final_y, card_w, card_h, final_title, final_sub, winner=True))

    canvas_w = col_x[-1] + card_w + padding
    svg = f'<svg width="{canvas_w}" height="{canvas_h}" style="position:absolute;left:0;top:0;">' + "".join(svg_lines) + '</svg>'

    html_doc = f'''
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  body {{
    margin:0; padding:0;
    background:#0b1020; color:#e8efff;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial;
  }}
  .wrap {{
    position:relative; width:{canvas_w}px; height:{canvas_h}px; margin:16px auto;
    border:1px solid #223055; border-radius:12px; background:#0b1020;
  }}
  .card {{
    position:absolute; border:1px solid; border-radius:12px; padding:8px 10px;
    box-sizing:border-box;
  }}
  .tok {{ font-weight:600; }}
  .sub {{ color:#94a3b8; font-size:12px; margin-top:2px; }}
  .header {{
    margin: 0 auto 8px auto; max-width:{canvas_w}px; padding: 8px 4px;
    color:#a6b6d6; font-size:13px;
  }}
  code {{
    background: rgba(255,255,255,0.06); padding:2px 6px; border-radius:6px; border:1px solid #223055;
  }}
</style>
</head>
<body>
  <div class="header">
    <b>Step {step}</b> &nbsp; seed_rt=<code>{seed_rt}</code> · m_layers={m_layers} · group_size={group_size} · g_dist={g_dist} · H={H} · nsec={nsec}
  </div>
  <div class="wrap">
    {svg}
    {''.join(cards_html)}
  </div>
</body>
</html>
'''
    return html_doc

# ---------- Streamlit UI ----------
st.sidebar.title("Load a Single Trace")
default_dir = st.sidebar.text_input("Default folder (optional)", value="tournament_traces")
filename = st.sidebar.text_input("JSON file name (e.g., step_001_tournament.json)", value="")
uploaded = st.sidebar.file_uploader("…or upload a JSON trace", type=["json"])
load_btn = st.sidebar.button("Render bracket")

st.title("Tournament Bracket (Connected)")
st.caption("One file at a time — visualizes layer-wise token sampling trace as a tournament structure.")

trace: Optional[Dict[str, Any]] = None

if load_btn:
    try:
        if uploaded is not None:
            trace = load_trace_from_uploader(uploaded)
        elif filename:
            path = filename if os.path.isabs(filename) else os.path.join(default_dir, filename)
            if not os.path.exists(path):
                st.error(f"File not found: {path}")
            else:
                trace = load_trace_from_path(path)
        else:
            st.warning("Enter a filename or upload a JSON file.")
    except Exception as e:
        st.error(f"Error loading trace: {e}")

# ✅ Prevent running build_html() if trace is None
if trace is None:
    st.info("Please load a valid JSON trace and click **Render bracket**.")
    st.stop()

# ---------- Render HTML Bracket ----------
try:
    doc = build_html(trace)
    st_html(doc, height=700, scrolling=True)
except Exception as e:
    st.error(f"Error while building HTML: {e}")
