import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from wss_model import *  # 物理核心（常数 / 风场 / 湍流 / 颗粒动力学）见 wss_model.py

# ============================================================
# Wind & Sand Simulator 2.4 · 极简界面版
# 教学原型：参数化水平风速场 + 湍流场 + 分状态沙粒运动（单向耦合）
# 2.1 修复 NumPy 2.x 崩溃；重建沙粒-植被耦合；重标定阻力/阈值/孔隙率
# 2.2 紧凑标签页 UI；逐颗粒运动状态（静止/跃移/悬移）；2D 横截面视窗
# 2.3 去重复视图；统一图高/标签页/侧栏标签；结果缓存加速参数回溯
# 2.4 物理核心独立为 wss_model.py 并加回归测试（model_checks.py）：
#     · 湍流强度改为逐点场（背景项随未扰动风速、植被项由冠层剪切层生成）
#     · 颗粒状态逐时步演化，跌入/穿出植被区会自动换态
#     · 新增植被枝叶拦截、床面再起动、Saffman 升力、壁面精确落点
#     · 积分步长可选（0.002/0.004/0.008 s），版本号在页面/导出/图内统一
# 视图：总览两幅主图——机理剖面（动量亏损带 + 分档风矢 + 冠层轮廓）与
#       「定量风场 + 近床面自适应放大」双栏；两图共用一套图元与图例
# ============================================================

# 全局唯一字体栈：页面 CSS 与 Plotly 图表共用，避免两套字体并存
UI_FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft YaHei', "
           "'PingFang SC', 'Hiragino Sans GB', Arial, sans-serif")
FIG_FONT_SIZE = 10   # 图表内所有文字（轴、图例、注释、悬停）统一一档
# 2.3：全站图高统一为三档，所有图共用，保证页面节奏规整
FIG_H_MAIN = 380     # 主视图：横截面 / 风场热力图 / 输沙敏感性
FIG_H_FIELD = 470    # 风场 + 近床面双视窗（上风速场热力图 + 下跃移层放大）
FIG_H_TRAJ = 470     # 双视窗轨迹图（上悬移层 + 下跃移层）
FIG_H_SIDE = 320     # 对照分析图（沿程风速 / 降低率 / 廓线 / 扫描）
FIG_H_HIST = 280     # 直方图

# 全局统一浅色图表主题：所有图共用，避免逐图重复设置、保证视觉一致
_THEME = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
    font=dict(family=UI_FONT, size=FIG_FONT_SIZE, color="#334155"),
    xaxis=dict(gridcolor="#EDF1F5", zerolinecolor="#EDF1F5", linecolor="#D9E1E8",
               showgrid=True, title_font=dict(size=FIG_FONT_SIZE)),
    yaxis=dict(gridcolor="#EDF1F5", zerolinecolor="#EDF1F5", linecolor="#D9E1E8",
               showgrid=True, title_font=dict(size=FIG_FONT_SIZE)),
    colorway=["#0D9488", "#3B82F6", "#E0A83B", "#16A34A", "#A16207", "#9333EA", "#E05C4B"],
    hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#D9E1E8",
                    font=dict(color="#334155", size=FIG_FONT_SIZE)),
    legend=dict(orientation="h", y=1.11, x=0, font=dict(size=FIG_FONT_SIZE),
                bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=8, r=8, t=22, b=8),
))
pio.templates["wss_light"] = _THEME
pio.templates.default = "wss_light"

st.set_page_config(page_title=f"Wind & Sand Simulator {APP_VERSION}", layout="wide",
                   initial_sidebar_state="expanded")


_UI_CSS = """
<style>
:root {
  --line:#E3E9EE; --muted:#7A8896; --ink:#1F2D3D; --accent:#0D9488;
  --ui-font:__UI_FONT__;
  --fs-body:.82rem;   /* 正文 / 说明 / 注解 / 公式 / 提示 / 列表：统一一档 */
  --fs-label:.72rem;  /* 分区标签 / KPI 标签 / 单位 / 角标：统一第二档 */
}
.block-container { padding-top:.95rem; padding-bottom:1.2rem; max-width:1440px; }
[data-testid="stHeader"] { height:2rem; background:transparent; }
[data-testid="stSidebar"] { background:#F7F9FB; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:.8rem; }
[data-testid="stVerticalBlock"] { gap:.6rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.4rem; }

/* ---- 字体统一：界面与说明文字共用一套无衬线字体 ----
   不加 !important、不覆盖 .katex 内部类，KaTeX 数学符号仍由其专用字体绘制。 */
html, body, .block-container, [data-testid="stAppViewContainer"],
[data-testid="stSidebar"], [data-testid="stMarkdownContainer"],
[data-testid="stCaptionContainer"], [data-testid="stAlert"],
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stExpander"], [data-baseweb="tab"],
h1, h2, h3, h4, p, li, label, button, input, textarea, select,
.stCaption, .mini-note, .formula-note, .sect, .app-head, .kpi, .state {
  font-family:var(--ui-font);
}

/* ---- 字号统一：所有说明性文字同一档 ---- */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
p, li, .stCaption, [data-testid="stCaptionContainer"],
[data-testid="stAlert"], .mini-note, .formula-note, .app-sub,
.stButton > button, .stDownloadButton > button,
[data-testid="stTabs"] [data-baseweb="tab"] {
  font-size:var(--fs-body) !important; line-height:1.5;
}
.mini-note, .formula-note { color:var(--muted); margin:.18rem 0 .42rem; }
/* 第二档：微标签统一 */
.sect, .app-badge, .kpi .k, .kpi .v small, .state .s-d, .state .s-t,
[data-testid="stMetricLabel"] { font-size:var(--fs-label) !important; }

/* ---- 公式：与正文同字号、左对齐、去掉 KaTeX 默认放大与居中留白 ---- */
[data-testid="stLatex"] { margin:.1rem 0 .28rem; overflow-x:auto; }
.katex { font-size:var(--fs-body) !important; }
.katex-display { margin:.2rem 0 !important; text-align:left !important; }
.katex-display > .katex { text-align:left !important; }
.katex .mtext { font-family:var(--ui-font); }  /* 公式内中文与界面同字体 */

hr { margin:.28rem 0 !important; border-color:var(--line) !important; }
h1, h2, h3 { letter-spacing:-.02em; }
h1 { font-size:1.5rem !important; font-weight:660 !important; }
h2 { font-size:1.0rem !important; font-weight:640 !important; padding-top:.1rem; padding-bottom:.2rem; border-bottom:1px solid var(--line); }
h3 { font-size:.88rem !important; font-weight:620 !important; }
[data-testid="stMetric"] { padding:.26rem .45rem .3rem; border:1px solid var(--line); border-radius:6px; }
[data-testid="stMetricLabel"] { color:var(--muted) !important; }
[data-testid="stMetricValue"] { font-size:1.1rem !important; }
[data-testid="stPlotlyChart"] { border:1px solid var(--line); border-radius:7px; padding:.12rem; background:#fff; }
.stButton > button, .stDownloadButton > button { border-radius:6px; min-height:2.05rem; font-weight:600; }
.stSlider, .stSelectSlider { padding-top:0; padding-bottom:0; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.1rem; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [data-baseweb="tab"] { padding:.32rem .75rem; font-weight:600; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:6px; }
[data-testid="stAlert"] { padding:.38rem .62rem; border-radius:6px; }
.app-head { display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; padding:.05rem 0 .5rem; border-bottom:1px solid var(--line); margin-bottom:.5rem; }
.app-title { font-size:1.4rem; font-weight:680; letter-spacing:-.035em; color:var(--ink); line-height:1.15; }
.app-sub { color:var(--muted); margin-top:.2rem; }
.app-badge { color:var(--accent); border:1px solid rgba(13,148,136,.35); background:rgba(13,148,136,.06); padding:.18rem .5rem; border-radius:20px; white-space:nowrap; }
/* 固定 4 列网格：8 张卡片排成整齐两行，避免窄屏挤压导致数值与单位换行重叠 */
.kpi-row { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.38rem; margin:.1rem 0 .55rem; }
.kpi { min-width:0; border:1px solid var(--line); border-radius:6px; padding:.3rem .5rem .34rem; background:#fff; overflow:hidden; }
.kpi .k { color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kpi .v { font-size:1.0rem; font-weight:660; color:var(--ink); line-height:1.28; white-space:nowrap; }
.kpi .v small { font-weight:500; color:var(--muted); margin-left:.15rem; }
.state-row { display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:.38rem; margin:.35rem 0 .45rem; }
.state { min-width:0; border:1px solid var(--line); border-left-width:3px; border-radius:6px; padding:.36rem .55rem; background:#fff; overflow:hidden; }
.state .s-t { font-weight:640; white-space:nowrap; }
.state .s-n { font-size:1.04rem; font-weight:680; }
.state .s-d { color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sect { text-transform:uppercase; letter-spacing:.11em; color:var(--muted); font-weight:680; margin:.7rem 0 .3rem; }
@media (max-width:900px) { .kpi-row, .state-row { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
/* 模型自检表：不依赖 pyarrow，直接用 HTML 渲染 */
.check-tbl { width:100%; border-collapse:collapse; margin:.15rem 0 .3rem; }
.check-tbl th { text-align:left; color:var(--muted); font-weight:640; padding:.26rem .42rem; border-bottom:1px solid var(--line); }
.check-tbl td { padding:.24rem .42rem; border-bottom:1px solid #F1F4F7; vertical-align:top; color:var(--ink); }
.check-tbl td.ok { color:#0F766E; font-weight:640; white-space:nowrap; }
.check-tbl td.bad { color:#B42318; font-weight:640; white-space:nowrap; }
/* ---- 轻量动效：只做入场与悬停反馈，遵循系统「减弱动效」设置 ----
   卡片与图表的内容只在真正更新时变化，因此入场动画不会在拖动滑块时反复播放。 */
@keyframes wssRise { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:none; } }
@keyframes wssPulse { 0%,100% { box-shadow:0 0 0 0 rgba(217,119,6,0); } 50% { box-shadow:0 0 0 3px rgba(217,119,6,.10); } }
@media (prefers-reduced-motion: no-preference) {
  .kpi, .state { animation:wssRise .32s cubic-bezier(.22,.61,.36,1) both; }
  .kpi:nth-child(2), .state:nth-child(2) { animation-delay:.03s; }
  .kpi:nth-child(3), .state:nth-child(3) { animation-delay:.06s; }
  .kpi:nth-child(4), .state:nth-child(4) { animation-delay:.09s; }
  .kpi:nth-child(5) { animation-delay:.12s; }
  .kpi:nth-child(6) { animation-delay:.15s; }
  .kpi:nth-child(7) { animation-delay:.18s; }
  .kpi:nth-child(8) { animation-delay:.21s; }

  [data-testid="stPlotlyChart"] { animation:wssRise .40s cubic-bezier(.22,.61,.36,1) both; }

  .kpi, .state { transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
  .kpi:hover, .state:hover { transform:translateY(-1px); box-shadow:0 5px 14px rgba(15,23,42,.07); border-color:#D6E0E8; }

  .stButton > button, .stDownloadButton > button { transition:transform .16s ease, box-shadow .16s ease; }
  .stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 5px 14px rgba(13,148,136,.18); }

  [data-testid="stAlert"] { animation:wssPulse 2.2s ease-in-out 2; }
  [data-testid="stTabs"] [data-baseweb="tab"] { transition:background-color .16s ease; }
  [data-testid="stTabs"] [data-baseweb="tab"]:hover { background-color:rgba(13,148,136,.06); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation:none !important; transition:none !important; }
}
</style>
"""
st.markdown(_UI_CSS.replace("__UI_FONT__", UI_FONT), unsafe_allow_html=True)


# ------------------------------------------------------------
# 6. 图表
# ------------------------------------------------------------

VEG_GREEN = "#268A4A"
BED_SAND = "#E4CEA3"
# 2.4：两幅主视图共用一套图元语言（沙层 / 树冠 / 风矢 / 场色阶），避免各画一套
BED_SAND_EDGE = "#D3BD92"
TRUNK_BROWN = "#A98A63"
INK_COLOR = "#1F2D3D"
MUTED_COLOR = "#7A8896"
# 风速场色阶：白（无风）→ 深青（来流）。色标在中高速段多设几档，
# 否则对数廓线让 5–7 m/s 占了整幅图的 80%，画面会糊成一片均匀的深青。
FIELD_COLORSCALE = [[0.00, "#F4FAF9"], [0.30, "#E8F3F0"], [0.55, "#C9E5DE"],
                    [0.72, "#A6D3C8"], [0.85, "#7CBCAF"], [0.94, "#54A296"],
                    [1.00, "#2E8377"]]


def _rgba(hex_color, alpha):
    """#RRGGBB → rgba(...)，用于需要透明度的填充与描边。"""
    h = hex_color.lstrip("#")
    return (f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},"
            f"{float(np.clip(alpha, 0.0, 1.0)):.2f})")


def _sample_paths(trajectories, mode, limit):
    """按等间隔抽样轨迹：保留状态内部差异，同时控制图元数量。"""
    picked = [t for t in trajectories if t[4] == mode]
    if limit and len(picked) > limit:
        idx = np.linspace(0, len(picked) - 1, limit).astype(int)
        picked = [picked[i] for i in idx]
    return picked


def _merge_paths(paths):
    """把多条轨迹拼成带 None 断点的单条线迹。"""
    px, py = [], []
    for xs, zs, *_ in paths:
        if xs:
            px.extend(xs + [None])
            py.extend(zs + [None])
    return px, py


def near_bed_range(paths, lo=0.04, hi=0.30, quantile=0.995, pad=1.35):
    """由跃移轨迹的实际高度定近床面放大窗的量程。

    固定 0–0.22 m 会让 3 cm 的弹跳只占七分之一高度，看起来仍是一条线；
    按本次轨迹的 99.5 分位乘 1.35 取整，既能包住最高的弹跳，也不留大片空白。
    """
    heights = [float(max(p[1])) for p in paths if p[1]]
    if not heights:
        return 0.10
    z = float(np.quantile(np.asarray(heights), quantile)) * pad
    return float(np.clip(np.ceil(z * 100.0) / 100.0, lo, hi))


def layer_envelope(paths, n_bins=140, quantile=0.90):
    """按 x 分箱统计跃移层上缘（分位数），得到可读的"输沙层厚度"包络。

    140 m × 0.05 m 的窗口里，每条 0.4 m 的弹跳只占 3–4 px 宽，单看轨迹必然是
    一排竖直细线。按 1 m 一箱取 90 分位高度并平滑，包络就能读出
    "植被内层变薄、下风逐渐恢复"的趋势，而单条轨迹仍保留作纹理。
    """
    xs, zs = [], []
    for px, pz, *_ in paths:
        xs.extend(px)
        zs.extend(pz)
    if not xs:
        return np.array([]), np.array([])
    xs = np.asarray(xs, dtype=float)
    zs = np.maximum(np.asarray(zs, dtype=float), 0.0)
    edges = np.linspace(0.0, DOMAIN_END, n_bins + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, n_bins - 1)
    env = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = idx == i
        if np.count_nonzero(sel) >= 4:
            env[i] = float(np.quantile(zs[sel], quantile))
    centers = 0.5 * (edges[1:] + edges[:-1])
    ok = np.isfinite(env)
    if np.count_nonzero(ok) >= 3:
        env = np.interp(centers, centers[ok], env[ok])
    return centers[ok], env[ok]


def _add_canopy(fig, x0, x1, height, p_eff, row=None, col=None, frost=False):
    """植被带冠层：平缓起伏的填充轮廓（成片绿篱）+ 细树干。

    比逐棵画三角形更接近成片植被的观感，且只有两条迹线，不制造视觉噪点。
    frost=True 时改用白色薄霜填充，用于深色风速场之上，保证轮廓仍可辨识。
    """
    loc = dict(row=row, col=col) if row is not None else {}
    width = max(float(x1 - x0), 1.0)
    n_wave = max(2, int(round(width / 8.0)))
    t = np.linspace(0.0, 1.0, 260)
    xs = x0 + t * width
    # 极扁的纵横比下，任何夸张起伏都会像装饰花纹；只留极缓的冠层轮廓
    top = height * (0.94 + 0.06 * np.sin(2.0 * np.pi * n_wave * t))
    base = height * 0.26
    fill = "rgba(255,255,255,0.26)" if frost else _rgba(VEG_GREEN, 0.10 + 0.10 * (1.0 - p_eff))
    edge = _rgba(VEG_GREEN, 0.80) if frost else _rgba(VEG_GREEN, 0.62)
    fig.add_trace(go.Scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([top, np.full_like(xs, base)]),
        mode="lines", fill="toself", fillcolor=fill,
        line=dict(color=edge, width=1.1),
        hoverinfo="skip", showlegend=False), **loc)
    n_trunk = int(np.clip(round(width / 4.2), 2, 16))
    trunk_x, trunk_y = [], []
    for xi in np.linspace(x0 + 1.5, x1 - 1.5, n_trunk):
        trunk_x += [float(xi), float(xi), None]
        trunk_y += [-0.02, base * 1.06, None]
    fig.add_trace(go.Scatter(x=trunk_x, y=trunk_y, mode="lines",
                             line=dict(color=TRUNK_BROWN, width=0.9), opacity=0.40,
                             hoverinfo="skip", showlegend=False), **loc)


def _add_wind_arrows(fig, u_ref, cover, porosity, veg_height, turbulence_intensity,
                     leaf_development, roughness, x_right, z_top, max_len=11.5):
    """风矢：长度与 local_wind 成正比，并按风速分档着色（浅＝弱风，深＝强风）。

    冠层内外的长度与颜色同时变淡，是这张剖面图的主信息——风被削弱了多少。
    按档分组绘制，既保留颜色编码，也避免为每支箭单独建迹线。
    """
    xs = [6.0, 22.0, 40.0, 56.0, 72.0, 90.0, 106.0]
    zs = [float(f) * z_top for f in (0.13, 0.34, 0.55, 0.76)]
    pts = [(x, z) for z in zs for x in xs]
    speeds = np.array([float(local_wind(x, z, u_ref, cover, porosity,
                                        float(veg_height), turbulence_intensity,
                                        leaf_development, roughness))
                       for x, z in pts])
    u_max = max(float(np.max(speeds)), 0.5)
    scale = max_len / u_max
    ramp = ["#B3C4D3", "#9EB2C4", "#879DB3", "#6F88A1", "#566F8A"]
    bins = [[] for _ in ramp]
    for (x, z), u in zip(pts, speeds):
        length = min(max(1.5, scale * u), x_right - 2.0 - x)
        if length < 1.6:
            continue
        idx = min(len(ramp) - 1, int(float(u) / u_max * len(ramp)))
        bins[idx].append((x, z, x + length * 0.80))
    for color, group in zip(ramp, bins):
        if not group:
            continue
        shaft_x, shaft_y = [], []
        for x0, z0, x1 in group:
            shaft_x += [x0, x1, None]
            shaft_y += [z0, z0, None]
        fig.add_trace(go.Scatter(x=shaft_x, y=shaft_y, mode="lines",
                                 line=dict(color=color, width=1.5),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[g[2] for g in group], y=[g[1] for g in group],
                                 mode="markers",
                                 marker=dict(symbol="triangle-right", size=7, color=color),
                                 hoverinfo="skip", showlegend=False))


def paths_to_xy(trajectories, mode=None, limit=None):
    """把若干条轨迹合并成带 None 断点的单条 Plotly 线迹坐标（旧调用入口）。"""
    picked = [t for t in trajectories if (mode is None or t[4] == mode)]
    if limit is not None:
        picked = picked[:limit]
    return _merge_paths(picked)


def legend_dummy(fig, items):
    """仅用于生成图例的占位迹，避免给示意图增加视觉噪点。"""
    for name, color, symbol in items:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(symbol=symbol, size=7, color=color),
                                 showlegend=True, hoverinfo="skip"))


def make_cross_section_figure(u_ref, cover, porosity, veg_height, trajectories,
                              pinned_x, deposited_x, soil_moisture_pct,
                              turbulence_intensity, leaf_development, roughness,
                              height=FIG_H_MAIN, intercepted_x=None, intercepted_z=None):
    """2D 横截面视窗：一眼看清「来流 — 植被减速 — 尾流恢复」的机理剖面。

    图元只有四层：动量亏损带（浅色等值面）、植被轮廓、风矢、床面颗粒。
    风矢长度与 local_wind 成正比，是局部风速的直观读数；浅色带为相对风速
    亏损 (1-U/U0) 的等值面，用来标出植被带及其下风侧「风被削弱」的范围。
    本图不承载定量的风速色标（见下方风速场图），因此不再叠画示意轨迹，
    以免与真实轨迹图产生两套互相矛盾的沙粒画法。
    """
    x_right = 120.0
    z_top = max(3.4, min(float(veg_height), 6.0) + 1.0)
    p_eff = effective_porosity(porosity, leaf_development)
    canopy_top = min(float(veg_height), z_top * 0.92)
    fig = go.Figure()

    # ---- 1. 动量亏损带：颜色越深＝风被削弱得越多（cover=0 时整片为白，自动隐去）
    if cover > 0:
        xg = np.linspace(0.0, x_right, 200)
        zg = np.linspace(0.02, z_top, 110)
        XG, ZG = np.meshgrid(xg, zg)
        u_loc = local_wind(XG, ZG, u_ref, cover, porosity, veg_height,
                           turbulence_intensity, leaf_development, roughness)
        u_bare, _ = wind_profile(ZG, u_ref, roughness)
        ratio = np.clip(u_loc / np.maximum(u_bare, 1e-9), 0.0, 1.0)
        # 色阶下限跟随本工况的最强亏损，弱植被与密植被都能看到分层，而不是一片白
        z_lo = float(np.clip(np.floor(float(np.min(ratio)) * 20.0) / 20.0, 0.25, 0.90))
        fig.add_trace(go.Contour(
            x=xg, y=zg, z=ratio, zmin=z_lo, zmax=1.0, showscale=False, hoverinfo="skip",
            contours=dict(coloring="heatmap", showlines=False,
                          start=z_lo, end=1.0, size=0.04),
            colorscale=[[0.0, "#57AFA1"], [0.40, "#9DCFC5"], [1.0, "#FFFFFF"]],
            opacity=0.62, line=dict(width=0)))

    # ---- 2. 床面：沙层 + 颗粒质感 + 含水结皮
    fig.add_shape(type="rect", x0=0, x1=x_right, y0=-0.46, y1=0,
                  fillcolor=BED_SAND, line=dict(width=0), layer="below")
    sand_rng = np.random.default_rng(11)
    n_grain = 260
    fig.add_trace(go.Scatter(
        x=sand_rng.uniform(0.5, x_right - 0.5, n_grain),
        y=sand_rng.uniform(-0.43, -0.03, n_grain), mode="markers",
        marker=dict(size=2.4, color="#CBB387"), opacity=0.55, hoverinfo="skip",
        showlegend=False))
    fig.add_shape(type="line", x0=0, x1=x_right, y0=0, y1=0,
                  line=dict(color=BED_SAND_EDGE, width=1))
    if soil_moisture_pct > 3.0:
        alpha = min(0.20 + soil_moisture_pct * 0.018, 0.62)
        fig.add_shape(type="rect", x0=0, x1=x_right, y0=-0.05, y1=0,
                      fillcolor=f"rgba(56,116,150,{alpha:.2f})", line=dict(width=0))

    # ---- 3. 植被带轮廓
    if cover > 0:
        _add_canopy(fig, VEG_START, VEG_END, canopy_top, p_eff)

    # ---- 4. 风矢（长度 ∝ 局部风速）
    _add_wind_arrows(fig, u_ref, cover, porosity, veg_height, turbulence_intensity,
                     leaf_development, roughness, x_right, z_top)

    # ---- 5. 床面颗粒状态：真实计算结果，不做示意轨迹
    if pinned_x is not None and len(pinned_x):
        px = np.asarray(pinned_x)
        px = px[::max(1, len(px) // 120)]
        fig.add_trace(go.Scatter(x=px, y=np.full(len(px), 0.035), mode="markers",
                                 marker=dict(symbol="diamond", size=4.5,
                                             color=STATE_COLORS[MOTION_PINNED]),
                                 hoverinfo="skip", showlegend=False))
    if deposited_x is not None and len(deposited_x):
        dx = np.asarray(deposited_x)
        fig.add_trace(go.Scatter(x=dx, y=np.full(len(dx), 0.035), mode="markers",
                                 marker=dict(symbol="x-thin", size=6,
                                             line=dict(width=1.5), color=DEPOSIT_COLOR),
                                 hoverinfo="skip", showlegend=False))
    # 被枝叶拦截的颗粒按捕获高度绘制，直接看出拦截发生在冠层内部哪一层
    if intercepted_x is not None and len(intercepted_x):
        ix = np.asarray(intercepted_x)
        iz = np.clip(np.asarray(intercepted_z, dtype=float), 0.02, z_top * 0.97)
        step = max(1, len(ix) // 90)
        fig.add_trace(go.Scatter(x=ix[::step], y=iz[::step], mode="markers",
                                 marker=dict(symbol="x-thin", size=5,
                                             line=dict(width=1.2), color=DEPOSIT_COLOR),
                                 opacity=0.75, hoverinfo="skip", showlegend=False))

    # ---- 6. 标注：只留三处关键文字，其余交给图元本身
    anno = dict(showarrow=False, font=dict(size=FIG_FONT_SIZE, color=MUTED_COLOR),
                yanchor="middle")
    fig.add_annotation(x=2, y=z_top * 0.965, text=f"来流 U₁₀ = {u_ref:.1f} m/s →",
                       xanchor="left", **anno)
    if cover > 0:
        fig.add_annotation(x=x_right - 2, y=z_top * 0.965, text="尾流恢复区",
                           xanchor="right", **anno)
        fig.add_annotation(x=(VEG_START + VEG_END) / 2, y=canopy_top + 0.30,
                           text=f"植被带 {VEG_START:.0f}–{VEG_END:.0f} m · 覆盖 {cover:.0%}",
                           showarrow=False, font=dict(size=FIG_FONT_SIZE, color=VEG_GREEN),
                           bgcolor="rgba(255,255,255,.86)", borderpad=3)
    else:
        fig.add_annotation(x=(VEG_START + VEG_END) / 2, y=z_top * 0.55,
                           text="无植被：全断面为裸地对数廓线", showarrow=False,
                           font=dict(size=FIG_FONT_SIZE, color="#9AA8B6"))
    if soil_moisture_pct > 3.0:
        fig.add_annotation(x=x_right - 2, y=-0.23, text=f"含水结皮 {soil_moisture_pct:.0f}%",
                           xanchor="right", showarrow=False,
                           font=dict(size=FIG_FONT_SIZE, color="#2E6F94"))

    fig.update_layout(height=height, margin=dict(l=54, r=16, t=30, b=46),
                      plot_bgcolor="#FFFFFF", paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False,
                      xaxis=dict(range=[0, x_right], showgrid=False, zeroline=False,
                                 title="水平距离 x (m)", showline=False),
                      yaxis=dict(range=[-0.46, z_top], showgrid=False, zeroline=False,
                                 title="高度 z (m)", showline=False, nticks=5))
    return fig


def make_trajectory_figure(u_ref, cover, soil_moisture_pct, trajectories,
                           pinned_x, deposited_x, state_counts,
                           intercepted_x=None, intercepted_z=None):
    """沙粒轨迹：默认双视窗（悬移层 / 近床面跃移层），单一状态时只保留一个窗。

    跃移弹跳高度只有几厘米，与悬移层共用纵轴会被压成一条线，故按高度分层；
    近床面窗的量程由本次轨迹的实际高度确定（见 near_bed_range）。当本工况没有
    悬移颗粒时，不再保留一个空的上半栏，改为单窗并标注悬移限。
    """
    susp_paths = [t for t in trajectories if t[4] == MOTION_SUSPENSION]
    salt_paths = [t for t in trajectories if t[4] == MOTION_SALTATION]
    z_bed = near_bed_range(salt_paths, hi=0.60)
    susp_heights = [max(t[1]) for t in susp_paths if t[1]]
    susp_top = max(0.8, min(4.0, (max(susp_heights) if susp_heights else 0.8) * 1.15))
    both = bool(susp_paths) and bool(salt_paths)

    if both:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.42, 0.58], vertical_spacing=0.16)
        fig.add_annotation(x=2.0, y=susp_top * 0.96, xanchor="left", yanchor="top",
                           text=f"悬移层 · 随流（0–{susp_top:.1f} m）", showarrow=False,
                           font=dict(size=FIG_FONT_SIZE, color="#475569"),
                           bgcolor="rgba(255,255,255,.86)", borderpad=3, row=1, col=1)
        fig.add_annotation(x=2.0, y=z_bed * 0.96, xanchor="left", yanchor="top",
                           text=f"近床面跃移层 · 弹跳 / 静止（0–{z_bed:.2f} m 放大）",
                           showarrow=False, font=dict(size=FIG_FONT_SIZE, color="#475569"),
                           bgcolor="rgba(255,255,255,.86)", borderpad=3, row=2, col=1)
    else:
        fig = go.Figure()

    def loc(row):
        return dict(row=row, col=1) if both else {}

    if both:
        ux, uy = paths_to_xy(trajectories, MOTION_SUSPENSION, 40)
        if ux:
            uy = [None if (v is not None and v > susp_top) else v for v in uy]
            fig.add_trace(go.Scatter(x=ux, y=uy, mode="lines",
                                     line=dict(color=STATE_COLORS[MOTION_SUSPENSION],
                                               width=1.5, dash="dot"),
                                     opacity=0.9, showlegend=False, hoverinfo="skip"),
                          **loc(1))

    row_salt = 2 if both else 1
    fig.add_hrect(y0=-0.02, y1=0, fillcolor=BED_SAND, line_width=0, layer="below",
                  **loc(row_salt))
    if soil_moisture_pct > 3.0:
        alpha = min(0.18 + soil_moisture_pct * 0.018, 0.66)
        fig.add_hrect(y0=-0.006, y1=0, fillcolor=f"rgba(45,110,150,{alpha:.2f})",
                      line_width=0, layer="below", **loc(row_salt))
    if cover > 0:
        fig.add_vrect(x0=VEG_START, x1=VEG_END, fillcolor=VEG_GREEN, opacity=0.09,
                      line_width=0, **loc(row_salt))

    sx, sy = paths_to_xy(trajectories, MOTION_SALTATION, 80)
    if sx:
        sy = [None if (v is not None and v > z_bed) else v for v in sy]
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
                                 line=dict(color=STATE_COLORS[MOTION_SALTATION], width=1.3),
                                 opacity=0.88, showlegend=False, hoverinfo="skip"),
                      **loc(row_salt))
    if len(pinned_x):
        px = np.asarray(pinned_x)
        px = px[::max(1, len(px) // 140)]
        fig.add_trace(go.Scatter(x=px, y=np.full(len(px), 0.02 * z_bed), mode="markers",
                                 marker=dict(symbol="diamond", size=5,
                                             color=STATE_COLORS[MOTION_PINNED]),
                                 showlegend=False, hoverinfo="skip"), **loc(row_salt))
    if len(deposited_x):
        dx = np.asarray(deposited_x)
        dx = dx[::max(1, len(dx) // 60)]
        fig.add_trace(go.Scatter(x=dx, y=np.full(len(dx), 0.02 * z_bed), mode="markers",
                                 marker=dict(symbol="x-thin", size=5,
                                             line=dict(width=1.1), color=DEPOSIT_COLOR),
                                 opacity=0.85, showlegend=False, hoverinfo="skip"),
                      **loc(row_salt))

    # 被枝叶拦截的颗粒：按捕获高度落到对应窗口，红色叉号即为拦截点
    if intercepted_x is not None and len(intercepted_x):
        ix = np.asarray(intercepted_x, dtype=float)
        iz = np.asarray(intercepted_z, dtype=float)
        for sel, row, top in ((iz <= z_bed, row_salt, z_bed),
                              (iz > z_bed, 1 if both else row_salt, susp_top)):
            if not np.any(sel):
                continue
            xs_i, zs_i = ix[sel], np.clip(iz[sel], 0.0, top * 0.98)
            step = max(1, len(xs_i) // 120)
            fig.add_trace(go.Scatter(x=xs_i[::step], y=zs_i[::step], mode="markers",
                                     marker=dict(symbol="x-thin", size=5,
                                                 line=dict(width=1.1), color=DEPOSIT_COLOR),
                                     opacity=0.7, showlegend=False, hoverinfo="skip"),
                          **loc(row))

    if not both:
        rng = z_bed if salt_paths else susp_top
        fig.add_annotation(x=0.0, y=rng, xanchor="left", yanchor="bottom",
                           text=("近床面跃移层 · 弹跳 / 静止" if salt_paths else "悬移层 · 随流")
                                + f"（0–{rng:.2f} m）",
                           showarrow=False, font=dict(size=FIG_FONT_SIZE, color="#475569"),
                           bgcolor="rgba(255,255,255,.86)", borderpad=3)
        if not susp_paths:
            fig.add_annotation(x=90.0, y=rng * 0.90, xanchor="right", text="本工况无悬移颗粒",
                               showarrow=False, font=dict(size=FIG_FONT_SIZE, color="#94A3B8"),
                               bgcolor="rgba(255,255,255,.86)", borderpad=3)

    def _n(count):
        # 0 个时写“—”而不是“0”：计数式图例里的零值只是噪音
        return str(count) if count else "—"

    n_intercept = len(intercepted_x) if intercepted_x is not None else 0
    legend_dummy(fig, [(f"跃移 ({_n(state_counts['saltation'])})",
                        STATE_COLORS[MOTION_SALTATION], "circle"),
                       (f"悬移 ({_n(state_counts['suspension'])})",
                        STATE_COLORS[MOTION_SUSPENSION], "circle"),
                       (f"静止 ({_n(state_counts['pinned'])})",
                        STATE_COLORS[MOTION_PINNED], "diamond"),
                       (f"沉积 ({_n(len(deposited_x))})", DEPOSIT_COLOR, "x"),
                       (f"拦截 ({_n(n_intercept)})", DEPOSIT_COLOR, "x-thin")])
    if not salt_paths:
        fig.add_annotation(x=45.0, y=(z_bed if both else susp_top) * 0.5,
                           text="本工况无跃移颗粒（静止或悬移）", showarrow=False,
                           font=dict(size=FIG_FONT_SIZE, color="#94A3B8"),
                           **loc(row_salt))
    if soil_moisture_pct > 3.0:
        fig.add_annotation(x=2, y=0.85 * z_bed, text=f"含水率 {soil_moisture_pct:.0f}%：起动阈值被抬升",
                           showarrow=False, xanchor="left",
                           font=dict(size=FIG_FONT_SIZE, color="#2E6F94"), **loc(row_salt))

    if both:
        fig.update_yaxes(range=[0, susp_top], title_text="z (m)", row=1, col=1,
                         showgrid=False, zeroline=False)
        fig.update_yaxes(range=[-0.06 * z_bed, z_bed * 1.05], title_text="z (m)",
                         row=2, col=1, showgrid=False, zeroline=False,
                         tickvals=[0, round(z_bed / 2, 3), round(z_bed, 3)])
        fig.update_xaxes(range=[0, 90], row=1, col=1, showticklabels=False)
        fig.update_xaxes(range=[0, 90], title_text="水平距离 x (m)", row=2, col=1)
        top = 1.12
    else:
        fig.update_yaxes(range=[-0.06 * z_bed, z_bed * 1.05] if salt_paths else [0, susp_top],
                         title_text="z (m)", showgrid=False, zeroline=False)
        fig.update_xaxes(range=[0, 90], title_text="水平距离 x (m)")
        top = 1.06

    fig.update_layout(height=FIG_H_TRAJ, margin=dict(l=6, r=6, t=48, b=6),
                      showlegend=True,
                      legend=dict(orientation="h", y=top, x=0, yanchor="bottom",
                                  font=dict(size=FIG_FONT_SIZE), bgcolor="rgba(0,0,0,0)"))
    return fig


def make_transport_wind_figure(u_ref, cover, porosity, veg_height, turbulence_intensity,
                               leaf_development, roughness, soil_moisture_pct,
                               diameters_um, particle_masses, z_probe=0.2,
                               downstream_x=80.0):
    """输沙能力随来流风速的响应：裸地参考 vs 植被下风侧。

    Bagnold 型关系近似 q ∝ u*³，因此两条曲线都是上凸的；植被的存在把整条
    曲线压低并推迟起沙风速，二者的间距就是该工况下植被的抑制量。
    """
    speeds = np.linspace(2.0, 20.0, 19)
    bare, veg = [], []
    for u in speeds:
        m = calculate_transport_metrics(u, cover, porosity, veg_height, turbulence_intensity,
                                        leaf_development, roughness, soil_moisture_pct,
                                        diameters_um, particle_masses,
                                        z_probe=z_probe, downstream_x=downstream_x)
        bare.append(m["q_bare"])
        veg.append(m["q_vegetated"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=speeds, y=bare, name="裸地参考",
                             line=dict(color="#94A3B8", width=2.0, dash="dash")))
    fig.add_trace(go.Scatter(x=speeds, y=veg, name=f"植被下风侧 x={downstream_x:.0f} m",
                             line=dict(color="#0D9488", width=2.6), fill="tozeroy",
                             fillcolor=_rgba("#0D9488", 0.10)))
    fig.add_vline(x=float(u_ref), line_dash="dot", line_color="#D97706", opacity=0.55)
    now = calculate_transport_metrics(u_ref, cover, porosity, veg_height, turbulence_intensity,
                                      leaf_development, roughness, soil_moisture_pct,
                                      diameters_um, particle_masses,
                                      z_probe=z_probe, downstream_x=downstream_x)
    fig.add_trace(go.Scatter(x=[float(u_ref)], y=[now["q_vegetated"]], mode="markers",
                             name="当前工况", showlegend=False,
                             marker=dict(size=9, color="#D97706",
                                         line=dict(color="#FFFFFF", width=1.6))))
    fig.add_annotation(x=float(u_ref), y=now["q_vegetated"], text=f"当前 {u_ref:.1f} m/s",
                       showarrow=True, arrowhead=0, ax=0, ay=-26,
                       font=dict(size=FIG_FONT_SIZE, color="#B45309"),
                       bgcolor="rgba(255,255,255,.88)", borderpad=3)
    fig.update_layout(height=FIG_H_SIDE, xaxis_title="来流 U₁₀ (m/s)",
                      yaxis_title="输沙能力 q (kg/(m·s))",
                      margin=dict(l=10, r=10, t=25, b=10),
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


def make_interception_curve(u_ref, porosity, veg_height, turbulence_intensity,
                            leaf_development, roughness, soil_moisture_pct,
                            diameters_um, particle_masses, current_cover,
                            downstream_x=80.0):
    """不同植被覆盖度下的粒径加权输沙能力削减率敏感性曲线。"""
    covers = np.linspace(0.0, 0.80, 17)
    values = []
    for c in covers:
        result = calculate_transport_metrics(u_ref, c, porosity, veg_height,
            turbulence_intensity, leaf_development, roughness, soil_moisture_pct,
            diameters_um, particle_masses, downstream_x=downstream_x)
        values.append(np.nan if result["reduction_pct"] is None else result["reduction_pct"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=covers * 100, y=values, mode="lines+markers",
                             name="模型削减率", line=dict(color="#0D9488", width=2.8)))
    current = calculate_transport_metrics(u_ref, current_cover, porosity, veg_height,
        turbulence_intensity, leaf_development, roughness, soil_moisture_pct,
        diameters_um, particle_masses, downstream_x=downstream_x)["reduction_pct"]
    if current is not None:
        fig.add_vline(x=current_cover * 100, line_dash="dash", line_color="#D97706",
                      annotation_text=f"当前 {current_cover:.0%}：{current:.1f}%")
    fig.update_layout(height=FIG_H_SIDE, xaxis_title="植被覆盖度（%）", yaxis_title="输沙能力削减率（%）",
                      yaxis=dict(rangemode="tozero"), margin=dict(l=10, r=10, t=25, b=10))
    return fig


def make_field_figure(u_ref, cover, porosity, veg_height, trajectories,
                      deposited_x=None, pinned_x=None, soil_moisture_pct=0.0,
                      turbulence_intensity=0.15, leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                      roughness=DEFAULT_ROUGHNESS):
    """水平风速场 U(x,z) + 近床面分类轨迹：上下两栏共用横轴、各用合适纵轴。

    上图是定量风速场（色标 m/s，白色等值线间隔 1 m/s）；下图把近床面
    0–0.2 m 单独放大——跃移弹跳高度只有约 0.1 m，放进数米高的纵轴必然被
    压成一条线，这也是旧版「轨迹糊在地面」的根本原因。两栏横轴同步，
    可以直接对读「哪一段风速被削弱」与「沙粒在哪一段起跳、哪一段停下」。
    """
    z_top = max(3.2, min(float(veg_height), 9.0) + 0.9)
    xg = np.linspace(0.0, DOMAIN_END, 280)
    zg = np.linspace(0.012, z_top, 130)
    X, Z = np.meshgrid(xg, zg)
    U = local_wind(X, Z, u_ref, cover, porosity, veg_height, turbulence_intensity,
                   leaf_development, roughness)
    color_max = max(float(np.max(U)), 1.0)

    salt_paths = _sample_paths(trajectories, MOTION_SALTATION, 42)
    susp_paths = _sample_paths(trajectories, MOTION_SUSPENSION, 20)
    z_bed = near_bed_range(salt_paths, quantile=0.98, pad=1.10)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.66, 0.34], vertical_spacing=0.075)

    # ---- 上栏：风速场 + 等值线（等值线把对数廓线的竖向梯度画出来）
    fig.add_trace(go.Heatmap(
        x=xg, y=zg, z=U, zmin=0, zmax=color_max, colorscale=FIELD_COLORSCALE,
        colorbar=dict(thickness=11, len=0.56, y=0.80, yanchor="middle", outlinewidth=0,
                      tickfont=dict(size=9), ticks="outside", ticklen=2, tickcolor="#B9C4CF",
                      title=dict(text="m/s", side="top", font=dict(size=9))),
        hovertemplate="x：%{x:.1f} m<br>z：%{y:.2f} m<br>U：%{z:.2f} m/s<extra></extra>",
        name="水平风速", showlegend=False), row=1, col=1)
    fig.add_trace(go.Contour(
        x=xg, y=zg, z=U, showscale=False, hoverinfo="skip", showlegend=False,
        contours=dict(coloring="none", showlines=True, showlabels=False,
                      start=2.0, end=color_max, size=2.0),
        line=dict(color="rgba(255,255,255,0.45)", width=0.6)), row=1, col=1)
    if cover > 0:
        _add_canopy(fig, VEG_START, VEG_END, min(float(veg_height), z_top * 0.95),
                    effective_porosity(porosity, leaf_development), row=1, col=1,
                    frost=True)

    # 悬移轨迹在风速场里本来就该看见，用白色描边压住底色保证可读
    if susp_paths:
        hx, hy = _merge_paths(susp_paths)
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines",
                                 line=dict(color="rgba(255,255,255,0.60)", width=3.4),
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines",
                                 line=dict(color=STATE_COLORS[MOTION_SUSPENSION],
                                           width=1.2, dash="dot"),
                                 opacity=0.95, hoverinfo="skip", showlegend=False),
                      row=1, col=1)

    # ---- 下栏：近床面放大，真实轨迹按状态分类
    # 先用浅底把下栏框成"放大窗"，避免与上栏的渐隐底色混成一片
    fig.add_shape(type="rect", x0=0, x1=DOMAIN_END, y0=-0.10 * z_bed, y1=z_bed * 1.05,
                  fillcolor="#F2F7F8", line=dict(width=0), layer="below",
                  row=2, col=1)
    fig.add_shape(type="rect", x0=0, x1=DOMAIN_END, y0=-0.10 * z_bed, y1=0,
                  fillcolor=BED_SAND, line=dict(width=0), layer="below",
                  row=2, col=1)
    if soil_moisture_pct > 3.0:
        alpha = min(0.20 + soil_moisture_pct * 0.018, 0.62)
        fig.add_shape(type="rect", x0=0, x1=DOMAIN_END, y0=-0.03 * z_bed, y1=0,
                      fillcolor=f"rgba(56,116,150,{alpha:.2f})", line=dict(width=0),
                      layer="below", row=2, col=1)
    if cover > 0:
        fig.add_vrect(x0=VEG_START, x1=VEG_END, fillcolor=VEG_GREEN, opacity=0.07,
                      line_width=0, layer="below", row=2, col=1)
    if salt_paths:
        sx, sy = _merge_paths(salt_paths)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
                                 line=dict(color=STATE_COLORS[MOTION_SALTATION], width=0.7),
                                 opacity=0.40, hoverinfo="skip", showlegend=False),
                      row=2, col=1)
        ex, ey = layer_envelope(salt_paths)
        if ex.size:
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                                     line=dict(color=STATE_COLORS[MOTION_SALTATION], width=1.4),
                                     fill="tozeroy",
                                     fillcolor=_rgba(STATE_COLORS[MOTION_SALTATION], 0.16),
                                     hoverinfo="skip", showlegend=False), row=2, col=1)
    if pinned_x is not None and len(pinned_x):
        px = np.asarray(pinned_x)
        px = px[::max(1, len(px) // 40)]
        fig.add_trace(go.Scatter(x=px, y=np.full(len(px), 0.004 * z_bed), mode="markers",
                                 marker=dict(symbol="diamond", size=4,
                                             color=STATE_COLORS[MOTION_PINNED]),
                                 hoverinfo="skip", showlegend=False), row=2, col=1)
    if deposited_x is not None and len(deposited_x):
        dx = np.asarray(deposited_x)
        dx = dx[::max(1, len(dx) // 40)]
        fig.add_trace(go.Scatter(x=dx, y=np.full(len(dx), -0.002 * z_bed), mode="markers",
                                 marker=dict(symbol="x-thin", size=5,
                                             line=dict(width=1.3), color=DEPOSIT_COLOR),
                                 hoverinfo="skip", showlegend=False), row=2, col=1)

    # 空状态提示：避免「下栏一片空白」被误读为绘图失败
    if not salt_paths:
        hint = "全部颗粒静止：有效摩阻风速未超过起动阈值" if pinned_x is not None and len(pinned_x) \
            else "本工况无跃移颗粒"
        fig.add_annotation(x=DOMAIN_END * 0.62, y=z_bed * 0.72, text=hint, showarrow=False,
                           font=dict(size=FIG_FONT_SIZE, color="#9AA8B6"), row=2, col=1)

    # 图例固定五项，避免不同参数下条目增减导致版面跳动
    legend_dummy(fig, [("跃移", STATE_COLORS[MOTION_SALTATION], "circle"),
                       ("悬移", STATE_COLORS[MOTION_SUSPENSION], "circle"),
                       ("静止", STATE_COLORS[MOTION_PINNED], "diamond"),
                       ("沉积", DEPOSIT_COLOR, "x"),
                       ("植被带", VEG_GREEN, "triangle-up")])

    # ---- 标注与坐标：只在图内留两处必要文字
    fig.add_annotation(x=2, y=z_top * 0.90, text="来流 →", showarrow=False, xanchor="left",
                       font=dict(size=FIG_FONT_SIZE, color=INK_COLOR),
                       bgcolor="rgba(255,255,255,.78)", borderpad=3)

    fig.update_yaxes(range=[0, z_top], title_text="z (m)", nticks=5,
                     showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[-0.10 * z_bed, z_bed * 1.05], title_text="近床面 z (m)",
                     tickvals=[0, round(z_bed / 2, 2), z_bed], tickformat=".2f",
                     showgrid=False, zeroline=False, row=2, col=1)
    fig.update_xaxes(range=[0, DOMAIN_END], showgrid=False, zeroline=False,
                     showticklabels=False, ticks="", row=1, col=1)
    fig.update_xaxes(range=[0, DOMAIN_END], showgrid=False, zeroline=False,
                     title="水平距离 x (m)", dtick=20, row=2, col=1)

    fig.update_layout(height=FIG_H_FIELD, margin=dict(l=64, r=62, t=54, b=46),
                      showlegend=True,
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0,
                                  font=dict(size=FIG_FONT_SIZE),
                                  bgcolor="rgba(0,0,0,0)", itemsizing="constant"))
    return fig


def make_height_comparison(u_ref, cover, porosity, turbulence_intensity=0.15,
                           leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                           roughness=DEFAULT_ROUGHNESS):
    """不同均匀单层植被高度下，植被中心的垂向风速廓线。"""
    z = np.linspace(0.0, 12.0, 181)
    x_center = (VEG_START + VEG_END) / 2
    fig = go.Figure()
    colors = ["#64748b", "#2563eb", "#0d9488", "#16a34a", "#a16207", "#9333ea"]
    for h, color in zip([0.5, 1.0, 2.0, 4.0, 6.0], colors):
        speed = local_wind(np.full_like(z, x_center), z, u_ref, cover, porosity, h,
                           turbulence_intensity, leaf_development, roughness)
        fig.add_trace(go.Scatter(x=speed, y=z, mode="lines", name=f"H={h:g} m",
                                 line=dict(color=color, width=2.4)))
    fig.add_hline(y=DEFAULT_VEG_HEIGHT, line_dash="dot", line_color="gray", opacity=0.4)
    fig.update_layout(height=FIG_H_SIDE, xaxis_title="水平风速 U (m/s)", yaxis_title="高度 z (m)",
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12, x=0))
    return fig


def make_ground_profile(u_ref, cover, porosity, veg_height, turbulence_intensity=0.15,
                        leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                        roughness=DEFAULT_ROUGHNESS):
    """不同近地表高度上的沿程风速；用于识别植被区减速和下风向恢复。"""
    x = np.linspace(0, DOMAIN_END, 300)
    z_levels = [0.0, 0.1, 0.2, 0.3, 0.6]
    fig = go.Figure()
    for zi in z_levels:
        speed = local_wind(x, zi, u_ref, cover, porosity, veg_height,
                           turbulence_intensity, leaf_development, roughness)
        fig.add_trace(go.Scatter(x=x, y=speed, mode="lines", name=f"z={zi:.1f} m"))
    fig.add_vrect(x0=VEG_START, x1=VEG_END, fillcolor="green", opacity=0.10,
                  line_width=0)
    fig.add_vline(x=VEG_END, line_dash="dot", line_color="gray")
    fig.update_layout(height=FIG_H_SIDE, xaxis_title="水平距离 x (m)", yaxis_title="水平风速 U (m/s)",
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12, x=0))
    return fig


def make_reduction_figure(u_ref, cover, porosity, veg_height, z_probe=0.2,
                          turbulence_intensity=0.15,
                          leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                          roughness=DEFAULT_ROUGHNESS):
    """指定高度的风速降低率沿程分布。"""
    x = np.linspace(0, DOMAIN_END, 300)
    bare, _ = wind_profile(np.full_like(x, z_probe), u_ref, roughness)
    vegetated = local_wind(x, z_probe, u_ref, cover, porosity, veg_height,
                           turbulence_intensity, leaf_development, roughness)
    reduction = np.maximum(0.0, (bare - vegetated) / np.maximum(bare, 1e-9) * 100.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=reduction, mode="lines", name="风速降低率",
                             line=dict(color="#0D9488", width=2.8), fill="tozeroy"))
    fig.add_vrect(x0=VEG_START, x1=VEG_END, fillcolor="green", opacity=0.10,
                  line_width=0)
    fig.add_vline(x=VEG_END, line_dash="dot", line_color="gray")
    fig.update_layout(height=FIG_H_SIDE, xaxis_title="水平距离 x (m)", yaxis_title="风速降低率 (%)",
                      yaxis=dict(rangemode="tozero"), margin=dict(l=10, r=10, t=20, b=10))
    return fig, x, reduction


def make_turbulence_figure(u_ref, cover, porosity, veg_height, turbulence_intensity,
                           leaf_development, roughness, z_probe=0.2):
    """湍流强度场：沿程分布 + 垂向廓线。

    左图对比近地表与冠层顶的沿程湍流强度——冠层内因剪切层与枝叶尾流而抬升，
    出植被带后随尾流衰减回落到来流值；右图给出冠层中心与下风 15 m 的垂向廓线，
    可看出脉动峰值位于冠层上部（约 0.8H）而非地面。
    """
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16)
    h_canopy = max(float(veg_height), 0.5)
    x = np.linspace(0.0, DOMAIN_END, 400)
    all_i = []
    for z, color, label in [(z_probe, "#0D9488", f"近地表 z={z_probe:.1f} m"),
                            (h_canopy, "#D97706", f"冠层顶 z={h_canopy:.1f} m")]:
        _, _, il = turbulence_scales(x, z, u_ref, cover, porosity, veg_height,
                                     turbulence_intensity, leaf_development, roughness)
        vals = np.asarray(il) * 100.0
        all_i.append(vals)
        fig.add_trace(go.Scatter(x=x, y=vals, name=label,
                                 line=dict(color=color, width=2.2)), row=1, col=1)
    all_i.append(np.array([turbulence_intensity * 100.0]))
    fig.add_trace(go.Scatter(x=[0.0, DOMAIN_END],
                             y=[turbulence_intensity * 100.0] * 2, name="来流设定 Iᵤ",
                             line=dict(color="#94A3B8", width=1.2, dash="dash"),
                             hoverinfo="skip"), row=1, col=1)
    fig.add_vrect(x0=VEG_START, x1=VEG_END, fillcolor=VEG_GREEN, opacity=0.09,
                  line_width=0, row=1, col=1)

    z_max = 2.0 * max(float(veg_height), 1.0)
    zz = np.linspace(0.02, z_max, 180)
    for xm, color, label in [((VEG_START + VEG_END) / 2, "#0D9488", "冠层中心"),
                             (VEG_END + 15.0, "#8FA6B8", "下风 15 m")]:
        _, _, il = turbulence_scales(np.full_like(zz, xm), zz, u_ref, cover, porosity,
                                     veg_height, turbulence_intensity,
                                     leaf_development, roughness)
        fig.add_trace(go.Scatter(x=np.asarray(il) * 100.0, y=zz, name=label,
                                 line=dict(color=color, width=2.0, dash="dot")),
                      row=1, col=2)
    fig.add_hline(y=float(veg_height), line=dict(color=VEG_GREEN, width=1, dash="dot"),
                  row=1, col=2)

    lab = dict(showarrow=False, xanchor="left", yanchor="top",
               font=dict(size=FIG_FONT_SIZE, color="#475569"),
               bgcolor="rgba(255,255,255,.86)", borderpad=3)
    fig.add_annotation(x=0.02, y=0.965, xref="x domain", yref="y domain",
                       text="沿程分布", **lab)
    fig.add_annotation(x=0.98, y=0.965, xref="x2 domain", yref="y2 domain",
                       text="垂向廓线", **dict(lab, xanchor="right"))
    fig.update_xaxes(title_text="水平距离 x (m)", row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="I (%)", row=1, col=1, showgrid=False,
                     range=[max(0.0, min(float(np.min(v)) for v in all_i) - 1.5),
                            max(float(np.max(v)) for v in all_i) + 1.5])
    fig.update_xaxes(title_text="I (%)", row=1, col=2, showgrid=False)
    fig.update_yaxes(title_text="高度 z (m)", row=1, col=2, showgrid=False)
    fig.update_layout(height=FIG_H_SIDE, margin=dict(l=52, r=14, t=54, b=44),
                      showlegend=True,
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0,
                                  font=dict(size=FIG_FONT_SIZE), bgcolor="rgba(0,0,0,0)"))
    return fig


def make_cover_effect_figure(u_ref, porosity, veg_height, z_probe=0.2,
                             turbulence_intensity=0.15,
                             leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                             roughness=DEFAULT_ROUGHNESS):
    """固定高度下比较 0~80% 覆盖度的沿程风速。"""
    x = np.linspace(0, DOMAIN_END, 300)
    fig = go.Figure()
    colors = ["#64748b", "#2563eb", "#0d9488", "#16a34a", "#a16207"]
    for pct, color in zip([0, 20, 40, 60, 80], colors):
        speed = local_wind(x, z_probe, u_ref, pct / 100.0, porosity, veg_height,
                           turbulence_intensity, leaf_development, roughness)
        fig.add_trace(go.Scatter(x=x, y=speed, mode="lines", name=f"{pct}%",
                                 line=dict(color=color, width=2.4)))
    fig.add_vrect(x0=VEG_START, x1=VEG_END, fillcolor="green", opacity=0.08, line_width=0)
    fig.update_layout(height=FIG_H_SIDE, xaxis_title="水平距离 x (m)",
                      yaxis_title=f"风速 U (m/s)，z={z_probe:.1f} m",
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=1.12, x=0))
    return fig


def make_porosity_sweep_figure(u_ref, cover, veg_height, turbulence_intensity, z_probe=0.2,
                               leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                               roughness=DEFAULT_ROUGHNESS):
    ps, shelter, center = porosity_sweep(u_ref, cover, veg_height, turbulence_intensity,
                                         z_probe, leaf_development, roughness)
    best_i = int(np.argmax(shelter))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ps * 100, y=shelter, mode="lines+markers", name="遮蔽积分",
                             line=dict(color="#0D9488", width=2.8)))
    fig.add_trace(go.Scatter(x=ps * 100, y=center, mode="lines", name="中心降低率",
                             line=dict(color="#D97706", width=2.0, dash="dot"), yaxis="y2"))
    best_value = None if cover <= 0 else float(ps[best_i])
    if best_value is not None:
        fig.add_vline(x=best_value * 100, line_dash="dash", line_color="#A32D2D",
                      annotation_text=f"模型最优点 {best_value*100:.0f}%")
    fig.update_layout(height=FIG_H_SIDE, xaxis_title="植被基础孔隙率（%）",
                      yaxis=dict(title="下风侧相对风速亏损积分（m）"),
                      yaxis2=dict(title="冠层中心降低率（%）", overlaying="y", side="right",
                                  showgrid=False),
                      margin=dict(l=10, r=10, t=25, b=10),
                      legend=dict(orientation="h", y=1.14, x=0))
    return fig, best_value, float(shelter[best_i])


# ------------------------------------------------------------
# 7. 界面
# ------------------------------------------------------------

st.markdown(f"""
<div class="app-head">
  <div>
    <div class="app-title">风沙物理模拟</div>
    <div class="app-sub"><span translate="no" lang="en">Wind &amp; Sand Simulator</span> · 参数化风场 · 湍流场 · 植被防风 · 沙粒运动　|　__VER__</div>
  </div>
  <div class="app-badge">教学原型 · 非 CFD</div>
</div>""".replace("__VER__", f"v{APP_VERSION}"), unsafe_allow_html=True)

# ---- 参数默认值与工况预设：键名与侧栏控件一一对应，便于一键回填 ----
PARAM_DEFAULTS = dict(
    u_ref=8.0, cover_pct=30, porosity_pct=40, veg_height=2.0, soil_moisture_pct=2.0,
    mean_diameter_um=200, turbulence_pct=15, roughness_mm=0.5, leaf_pct=75,
    diameter_std_um=40, diameter_min_um=80, diameter_max_um=320,
    n_particles=500, random_seed=7, dt_ms=4, include_lift=True,
    z_probe=0.2, downstream_x=80.0,
)
# 四组典型工况：覆盖度/孔隙率/高度分别跨一个量级，便于课堂对照
PRESETS = {
    "裸地对照": dict(u_ref=8.0, cover_pct=0, porosity_pct=40, veg_height=2.0,
                    soil_moisture_pct=2.0, mean_diameter_um=200, turbulence_pct=15),
    "稀疏灌丛": dict(u_ref=8.0, cover_pct=20, porosity_pct=60, veg_height=1.0,
                    soil_moisture_pct=3.0, mean_diameter_um=200, turbulence_pct=20),
    "密林带": dict(u_ref=10.0, cover_pct=70, porosity_pct=30, veg_height=5.0,
                  soil_moisture_pct=2.0, mean_diameter_um=180, turbulence_pct=15),
    "强沙尘暴": dict(u_ref=18.0, cover_pct=10, porosity_pct=50, veg_height=1.5,
                   soil_moisture_pct=0.0, mean_diameter_um=140, turbulence_pct=35),
}

# URL 短键 -> (控件键, 下限, 上限, 步长)：用于分享/复现某一组参数
_URL_SPEC = {
    "u": ("u_ref", 2.0, 20.0, 0.5), "c": ("cover_pct", 0, 80, 5),
    "p": ("porosity_pct", 0, 80, 5), "h": ("veg_height", 0.5, 10.0, 0.5),
    "m": ("soil_moisture_pct", 0.0, 20.0, 0.5), "d": ("mean_diameter_um", 60, 400, 10),
    "ti": ("turbulence_pct", 5, 40, 1), "z0": ("roughness_mm", 0.1, 20.0, 0.1),
    "leaf": ("leaf_pct", 0, 100, 5), "sd": ("random_seed", 1, 99, 1),
    "n": ("n_particles", 100, 1000, 100), "dt": ("dt_ms", 2, 8, 2),
    "nx": ("downstream_x", 66.0, 130.0, 1.0),
}


def _snap(value, lo, hi, step):
    """把 URL 里的数值夹到量程内并对齐步长，避免直接 SetValue 越界。"""
    v = min(max(float(value), float(lo)), float(hi))
    snapped = round((v - lo) / step) * step + lo
    return round(snapped, 4)


def _apply_params(values):
    """预设/重置：写回控件状态并请求重算一次。"""
    for key, val in values.items():
        st.session_state[key] = val
    st.session_state["pending_run"] = True


def _load_from_url():
    """首次打开带参数的链接时，把 URL 值作为控件初值。"""
    for q_key, (widget_key, lo, hi, step) in _URL_SPEC.items():
        if widget_key in st.session_state:
            continue
        raw = st.query_params.get(q_key)
        if raw is None:
            continue
        try:
            st.session_state[widget_key] = _snap(raw, lo, hi, step)
        except (TypeError, ValueError):
            continue
    if "include_lift" not in st.session_state:
        lift = st.query_params.get("lift")
        if lift is not None:
            st.session_state["include_lift"] = lift in ("1", "true", "True")


def _sync_url():
    """把当前控件值写回 URL，便于复制链接复现同一工况。"""
    wanted = {}
    for q_key, (widget_key, *_rest) in _URL_SPEC.items():
        if widget_key in st.session_state:
            wanted[q_key] = str(st.session_state[widget_key])
    wanted["lift"] = "1" if st.session_state.get("include_lift", True) else "0"
    try:
        if st.query_params.to_dict() != wanted:
            st.query_params.from_dict(wanted)
    except Exception:      # 旧版 Streamlit 或非浏览器环境下忽略即可
        pass


_load_from_url()
# 控件默认值统一由 session_state 提供：这样预设/链接回填后不会与 widget 的
# default 冲突（Streamlit 会为此打印 "created with a default value..." 警告）。
for _key, _value in PARAM_DEFAULTS.items():
    st.session_state.setdefault(_key, _value)

PRESET_OPTIONS = ["裸地对照", "稀疏灌丛", "密林带", "强沙尘暴", "恢复默认参数"]
PRESET_NOTES = {
    "裸地对照": "覆盖 0% · 无植被，作为对照基准",
    "稀疏灌丛": "覆盖 20% · 孔隙 60% · H 1 m · Iᵤ 20%",
    "密林带": "覆盖 70% · 孔隙 30% · H 5 m · U₁₀ 10 m/s",
    "强沙尘暴": "U₁₀ 18 m/s · H 1.5 m · Iᵤ 35% · 粒径 140 μm",
    "恢复默认参数": "回到课程默认工况（覆盖 30% · H 2 m · U₁₀ 8 m/s）",
}


def _on_preset_change():
    """选中预设即应用并重算；随后清空选择，便于再次选同一项。"""
    choice = st.session_state.get("preset_select")
    if not choice:
        return
    if choice == "恢复默认参数":
        _apply_params(PARAM_DEFAULTS)
        st.session_state["preset_toast"] = "已恢复默认参数"
    else:
        _apply_params(dict(PARAM_DEFAULTS, **PRESETS[choice]))
        st.session_state["preset_toast"] = f"已应用预设：{choice}"
    st.session_state["preset_note"] = PRESET_NOTES.get(choice, "")
    st.session_state["preset_select"] = None
    snapshot = PARAM_DEFAULTS if choice == "恢复默认参数" else dict(PARAM_DEFAULTS, **PRESETS[choice])
    st.session_state["preset_snapshot"] = dict(snapshot)


with st.sidebar:
    st.markdown("<div class='sect'>工况预设</div>", unsafe_allow_html=True)
    st.selectbox("工况预设", PRESET_OPTIONS, index=None, key="preset_select",
                 placeholder="选择预设工况（选中即重算）", label_visibility="collapsed",
                 on_change=_on_preset_change)
    _note = st.session_state.get("preset_note")
    if _note:
        _snap = st.session_state.get("preset_snapshot", {})
        _drift = any(abs(float(st.session_state.get(k, v)) - float(v)) > 1e-9
                     for k, v in _snap.items()
                     if isinstance(v, (int, float)) and k in st.session_state)
        st.caption(_note + ("（已手动修改）" if _drift else ""))

    st.markdown("<div class='sect'>主控参数</div>", unsafe_allow_html=True)
    u_ref = st.slider("参考风速 U₁₀（m/s）", 2.0, 20.0, step=0.5, key="u_ref")
    cover_pct = st.slider("植被覆盖度（%）", 0, 80, step=5, key="cover_pct")
    porosity_pct = st.slider("基础孔隙率（%）", 0, 80, step=5, key="porosity_pct")
    veg_height = st.slider("植被高度 H（m）", 0.5, 10.0, step=0.5, key="veg_height")
    soil_moisture_pct = st.slider("土壤体积含水率（%）", 0.0, 20.0, step=0.5,
                                  key="soil_moisture_pct")
    mean_diameter_um = st.slider("平均粒径 μ（μm）", 60, 400, step=10,
                                 key="mean_diameter_um")
    turbulence_pct = st.slider("湍流强度 Iᵤ（%）", 5, 40, step=1, key="turbulence_pct")
    turbulence_intensity = turbulence_pct / 100.0
    with st.expander("精细参数"):
        roughness_mm = st.slider("粗糙度长度 z₀（mm）", 0.1, 20.0, step=0.1,
                                 key="roughness_mm")
        leaf_pct = st.slider("展叶率（%）", 0, 100, step=5, key="leaf_pct")
        diameter_std_um = st.slider("粒径标准差 σ（μm）", 10, 100, step=5,
                                    key="diameter_std_um")
        diameter_min_um = st.slider("最小粒径（μm）", 50, 200, step=10,
                                    key="diameter_min_um")
        diameter_max_um = st.slider("最大粒径（μm）", 200, 600, step=10,
                                    key="diameter_max_um")
        n_particles = st.select_slider(
            "代表性粒子数", options=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            key="n_particles")
        random_seed = st.slider("随机种子", 1, 99, step=1, key="random_seed")
        dt_ms = st.select_slider(
            "积分步长 Δt（ms）", options=[int(round(v * 1000)) for v in DT_OPTIONS],
            format_func=lambda v: f"{v} ms", key="dt_ms")
        include_lift = st.checkbox("计入 Saffman 气动升力", key="include_lift")
    roughness = roughness_mm / 1000.0
    leaf_development = leaf_pct / 100.0
    dt = dt_ms / 1000.0
    st.markdown("<div class='sect'>分析设置</div>", unsafe_allow_html=True)
    z_probe = st.select_slider("近地表分析高度 z（m）", options=[0.1, 0.2, 0.3, 0.5],
                               key="z_probe")
    downstream_x = st.slider("输沙评价截面 x（m）", 66.0, 130.0, step=1.0,
                             key="downstream_x")
    run = st.button("运行 / 更新模拟", type="primary", width="stretch")
    if not (diameter_min_um < mean_diameter_um < diameter_max_um):
        st.warning("需满足：最小粒径 < 平均粒径 < 最大粒径")
    st.markdown("<div class='mini-note'>预设/重置会自动重算；改单个滑块后需点「运行」。</div>",
                unsafe_allow_html=True)

_sync_url()

if _toast_msg := st.session_state.pop("preset_toast", None):
    st.toast(_toast_msg, icon="✅")


cover = cover_pct / 100.0
porosity = porosity_pct / 100.0
current_params = (u_ref, cover, porosity, veg_height, mean_diameter_um, diameter_std_um,
                  diameter_min_um, diameter_max_um, n_particles, turbulence_intensity,
                  soil_moisture_pct, leaf_development, roughness, random_seed,
                  dt, bool(include_lift))
auto_run = bool(st.session_state.pop("pending_run", False))
if "sim_params" not in st.session_state or run or auto_run:
    if not (diameter_min_um < mean_diameter_um < diameter_max_um):
        st.stop()
    st.session_state.sim_result = simulate_saltation(
        u_ref, cover, mean_diameter_um, n_particles, porosity, veg_height,
        turbulence_intensity, seed=random_seed, soil_moisture_pct=soil_moisture_pct,
        leaf_development=leaf_development, roughness=roughness,
        diameter_std_um=diameter_std_um, diameter_min_um=diameter_min_um,
        diameter_max_um=diameter_max_um, dt=dt, include_lift=include_lift)
    st.session_state.sim_params = current_params

saved_params = st.session_state.sim_params
if saved_params != current_params:
    st.info("参数已改变 — 点击左侧「运行 / 更新模拟」刷新结果。")
(old_u, old_cover, old_porosity, old_height, old_mean_d, old_std_d,
 old_min_d, old_max_d, old_n, old_ti, old_moisture, old_leaf, old_roughness,
 old_seed, old_dt, old_lift) = saved_params

sim = st.session_state.sim_result
trajectories = sim["trajectories"]
hop_lengths = sim["hop_lengths"]
u_star = sim["u_star"]
threshold = sim["threshold"]
particle_diameters = sim["diameters_um"]
particle_masses = sim["masses"]
deposited_x = sim["deposited_x"]
pinned_x = sim["pinned_x"]
counts = sim["state_counts"]
susp_limit = sim["suspension_limit"]

p_eff = effective_porosity(old_porosity, old_leaf)
effective_u_star = u_star * roughness_partition(old_roughness)
transport = calculate_transport_metrics(old_u, old_cover, old_porosity, old_height,
    old_ti, old_leaf, old_roughness, old_moisture, particle_diameters, particle_masses,
    z_probe=z_probe, downstream_x=downstream_x)
recovery = recovery_distance(old_u, old_cover, old_porosity, old_height, z_probe=z_probe,
                             turbulence_intensity=old_ti, leaf_development=old_leaf,
                             roughness=old_roughness)
base_speed, _ = wind_profile(np.array([z_probe]), old_u, old_roughness)
center_speed = local_wind((VEG_START + VEG_END) / 2, z_probe, old_u, old_cover, old_porosity,
                          old_height, old_ti, old_leaf, old_roughness)
reduction_center = ((float(base_speed[0] - center_speed) / float(base_speed[0]) * 100.0)
                    if base_speed[0] > 0 else 0.0)
in_belt = int(np.count_nonzero((deposited_x >= VEG_START) & (deposited_x <= VEG_END)))
# 湍流强度对比：冠层内 vs 下风 20 m vs 来流
ti_canopy = float(np.asarray(turbulence_scales(
    (VEG_START + VEG_END) / 2, z_probe, old_u, old_cover, old_porosity, old_height,
    old_ti, old_leaf, old_roughness)[2]))
ti_wake = float(np.asarray(turbulence_scales(
    VEG_END + 20.0, z_probe, old_u, old_cover, old_porosity, old_height,
    old_ti, old_leaf, old_roughness)[2]))


def show(fig, container=st, **kwargs):
    # 图例统一收口：旧版各图把 legend 放在 y=1.09~1.145（绘图区外上方），
    # 而 margin.t 只有 20~30px，窄列中换行后会压回曲线与 section 标题。
    # 这里统一改为「底边贴绘图区上沿、向顶部留白内排布」，并留足上边距。
    # 多迹线且带 name 时 Plotly 会默认显示图例（showlegend 为 None，不是 True），
    # 单迹线直方图等则不会显示，故按「实际会渲染图例」判断。
    legend_traces = [t for t in fig.data
                     if getattr(t, "name", None) and getattr(t, "showlegend", None) is not False]
    has_legend = (fig.layout.showlegend is True
                  or (fig.layout.showlegend is not False and len(legend_traces) > 1))
    m = fig.layout.margin
    # 各图原先 margin 只有 6~10px，轴标题会压到刻度数字上；统一留足四周留白。
    layout_kw = dict(margin=dict(l=max(m.l or 8, 50), r=max(m.r or 8, 12),
                                 b=max(m.b or 8, 44),
                                 t=max(m.t or 0, 44 if has_legend else 24)))
    if has_legend:
        layout_kw.update(
            showlegend=True,
            legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom",
                        font=dict(size=FIG_FONT_SIZE),
                        bgcolor="rgba(0,0,0,0)", itemsizing="constant"),
        )
    fig.update_layout(**layout_kw)
    container.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, **kwargs)


def section(text, container=st):
    container.markdown(f"<div class='sect'>{text}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner="正在运行模型自检…")
def model_check_rows():
    """界面内只跑快速档；全量档用 `python model_checks.py`。"""
    from model_checks import collect_checks
    return collect_checks(fast=True)


def kpi(items):
    cells = "".join(
        f"<div class='kpi'><div class='k'>{k}</div><div class='v'>{v}<small>{u}</small></div></div>"
        for k, v, u in items)
    return f"<div class='kpi-row'>{cells}</div>"


kpi_html = kpi([
    ("来流 U₁₀", f"{old_u:.1f}", "m/s"),
    ("植被覆盖度", f"{old_cover:.0%}", ""),
    ("有效孔隙率", f"{p_eff:.0%}", ""),
    ("植被高度", f"{old_height:.1f}", "m"),
    ("土壤含水率", f"{old_moisture:.1f}", "%"),
    ("平均粒径", f"{np.mean(particle_diameters):.0f}", "μm"),
    ("有效摩阻 u*", f"{effective_u_star:.3f}", "m/s"),
    ("粒子 / 种子", f"{old_n} / {old_seed}", ""),
])
state_html = f"""
<div class="state-row">
  <div class="state" style="border-left-color:{STATE_COLORS[MOTION_SALTATION]}">
    <div class="s-t">跃移 · 弹跳</div><div class="s-n">{counts['saltation']}</div>
    <div class="s-d">溅射维持 · 抛物线轨迹</div></div>
  <div class="state" style="border-left-color:{STATE_COLORS[MOTION_SUSPENSION]}">
    <div class="s-t">悬移 · 随流</div><div class="s-n">{counts['suspension']}</div>
    <div class="s-d">w_s &lt; {susp_limit:.2f} m/s</div></div>
  <div class="state" style="border-left-color:{STATE_COLORS[MOTION_PINNED]}">
    <div class="s-t">静止 · 被压住</div><div class="s-n">{counts['pinned']}</div>
    <div class="s-d">含水率 / 黏聚抬升阈值</div></div>
  <div class="state" style="border-left-color:{DEPOSIT_COLOR}">
    <div class="s-t">沉积</div><div class="s-n">{len(deposited_x)}</div>
    <div class="s-d">带内 {in_belt}（拦截 {sim['intercepted']}）</div></div>
</div>"""

tabs = st.tabs(["总览", "沙粒运动", "风场与防风", "输沙能力", "说明与导出"])
t_main, t_sand, t_wind, t_trans, t_doc = tabs

# ------------------------------------------------------------
# Tab 1 · 总览：KPI + 状态卡 + 2D 横截面视窗 + 风速场热力图
# ------------------------------------------------------------
t_main.markdown(kpi_html, unsafe_allow_html=True)
t_main.markdown(state_html, unsafe_allow_html=True)
section("2D 横截面视窗 · 机理剖面", t_main)
show(make_cross_section_figure(old_u, old_cover, old_porosity, old_height, trajectories,
                               pinned_x, deposited_x, old_moisture, old_ti, old_leaf,
                               old_roughness, intercepted_x=sim["intercepted_x"],
                               intercepted_z=sim["intercepted_z"]), t_main)
section("水平风速场 + 分类沙粒轨迹", t_main)
show(make_field_figure(old_u, old_cover, old_porosity, old_height, trajectories,
                       deposited_x, pinned_x, old_moisture, old_ti, old_leaf,
                       old_roughness), t_main)
t_main.markdown("<div class='mini-note'>箭头长短与颜色＝局部风速；浅青带＝风速亏损区；红叉＝被枝叶拦截（按捕获高度）。"
                "风速场下图按实际弹跳高度放大近床面，否则几厘米的跃移会被压成一条线。"
                f"本次 {old_n} 粒：拦截 {sim['intercepted']}、床面沉积 {len(deposited_x)}；"
                f"Δt {old_dt*1000:.0f} ms，{'含' if old_lift else '不含'}升力。</div>",
                unsafe_allow_html=True)

# ------------------------------------------------------------
# Tab 2 · 沙粒运动：双视窗轨迹 + 跃移/沉积/粒径统计
# ------------------------------------------------------------
section("沙粒运动轨迹 · 悬移层 / 近床面跃移层双视窗", t_sand)
show(make_trajectory_figure(old_u, old_cover, old_moisture, trajectories,
                            pinned_x, deposited_x, counts,
                            intercepted_x=sim["intercepted_x"],
                            intercepted_z=sim["intercepted_z"]), t_sand)
if effective_u_star <= threshold:
    t_sand.warning("有效摩阻风速低于含水率/粒径修正后的临界值：本工况以静止颗粒为主。")

h1, h2 = t_sand.columns(2, gap="small")
if hop_lengths:
    hist = go.Figure(go.Histogram(x=hop_lengths, nbinsx=18,
                                  marker_color=STATE_COLORS[MOTION_SALTATION]))
    hist.update_layout(height=FIG_H_HIST, xaxis_title="首次跃移水平距离 (m)", yaxis_title="粒子数")
    show(hist, h1)
    h1.markdown(f"<div class='mini-note'>平均首次跃移 <b>{np.mean(hop_lengths):.2f} m</b>，"
                f"记录到碰撞 {len(hop_lengths)} / {old_n} 个粒子。</div>", unsafe_allow_html=True)
else:
    h1.info("本工况无跃移碰撞记录：颗粒被压住不动或进入悬移。")
if len(deposited_x):
    dep = go.Figure(go.Histogram(x=deposited_x, nbinsx=24, marker_color=DEPOSIT_COLOR))
    dep.update_layout(height=FIG_H_HIST, xaxis_title="沉积位置 x (m)", yaxis_title="粒子数")
    dep.add_vrect(x0=VEG_START, x1=VEG_END, fillcolor="green", opacity=0.10, line_width=0)
    show(dep, h2)
else:
    h2.info("本工况未出现沉积：粒子维持运动或已移出计算域。")
dist_fig = go.Figure(go.Histogram(x=particle_diameters, nbinsx=18, marker_color="#0D9488"))
dist_fig.update_layout(height=FIG_H_HIST, xaxis_title="代表性颗粒直径 (μm)", yaxis_title="粒子数")
section("本次抽样的截断正态粒径分布", t_sand)
show(dist_fig, t_sand)
t_sand.markdown(f"<div class='mini-note'>样本均值 {np.mean(particle_diameters):.1f} μm（σ="
                f"{np.std(particle_diameters, ddof=1):.1f}），中位沉降速度 {sim['settling_median']:.2f} m/s，"
                f"悬移限 {susp_limit:.2f} m/s。状态每 0.016 s 按局地条件重判：颗粒被阵风抬升、"
                "进入冠层后可由跃移转悬移或被拦截，落到床面后又可能再起动——轨迹换色处即状态转换点。"
                "想看悬移：U₁₀ ≥14 m/s 且 d ≤100 μm；想看全静止：含水率 ≥12%。</div>",
                unsafe_allow_html=True)

# ------------------------------------------------------------
# Tab 3 · 风场与防风效应
# ------------------------------------------------------------
t_wind.markdown(kpi([
    ("中心风速降低率", f"{reduction_center:.1f}", "%"),
    ("摩阻风速 u*", f"{u_star:.3f}", "m/s"),
    ("湿度修正阈值", f"{threshold:.3f}", "m/s"),
    ("恢复距离 (90%)", "—" if recovery is None else f"{recovery:.1f}", "m"),
    ("冠层湍流强度", f"{ti_canopy:.0%}", ""),
    ("尾流湍流强度", f"{ti_wake:.0%}", ""),
    ("尾流特征长度 Lw",
     f"{wake_length_scale(old_u, old_cover, old_porosity, old_height, old_ti, old_leaf):.1f}", "m"),
    ("粗糙元分配系数", f"{roughness_partition(old_roughness):.2f}", ""),
]), unsafe_allow_html=True)
w1, w2 = t_wind.columns(2, gap="small")
section("近地表沿程风速（多高度）", w1)
show(make_ground_profile(old_u, old_cover, old_porosity, old_height,
                         old_ti, old_leaf, old_roughness), w1)
section("风速降低率沿程分布", w2)
red_fig, _, _ = make_reduction_figure(old_u, old_cover, old_porosity, old_height,
    z_probe=z_probe, turbulence_intensity=old_ti, leaf_development=old_leaf,
    roughness=old_roughness)
show(red_fig, w2)
w3, w4 = t_wind.columns(2, gap="small")
section("覆盖度对照（沿程风速）", w3)
cover_fig = make_cover_effect_figure(old_u, old_porosity, old_height, z_probe=z_probe,
    turbulence_intensity=old_ti, leaf_development=old_leaf, roughness=old_roughness)
show(cover_fig, w3)
section("植被高度 → 垂向风速廓线", w4)
height_fig = make_height_comparison(old_u, old_cover, old_porosity, old_ti,
                                    old_leaf, old_roughness)
show(height_fig, w4)
section("湍流强度 · 植被区 / 尾流区 / 来流", t_wind)
show(make_turbulence_figure(old_u, old_cover, old_porosity, old_height,
                            old_ti, old_leaf, old_roughness, z_probe=z_probe), t_wind)
t_wind.markdown("<div class='mini-note'>I 逐点计算：背景项（按未扰动风速标定）＋冠层剪切层/枝叶尾流项（峰值约 0.8H），"
                "出带后按尾流长度衰减，故冠层内 &gt; 尾流区 &gt; 来流。右图绿点线为冠层顶。</div>",
                unsafe_allow_html=True)
section("孔隙率扫描 · 较优疏透度", t_wind)
por_fig, best_porosity, _ = make_porosity_sweep_figure(
    old_u, old_cover, old_height, old_ti, z_probe, old_leaf, old_roughness)
show(por_fig, t_wind)
t_wind.markdown("<div class='mini-note'>恢复距离＝植被带末端到亏损恢复 90% 的距离（域内 140 m）。"
                f"模型较优孔隙率：<b>{'不适用（覆盖度 0%）' if best_porosity is None else format(best_porosity, '.0%')}</b>。"
                "绿线＝下风 60 m 亏损积分，橙虚线＝冠层中心降低率（右轴）；单峰为经验响应，非林带设计值。</div>",
                unsafe_allow_html=True)

# ------------------------------------------------------------
# Tab 4 · 输沙能力
# ------------------------------------------------------------
reduction_pct = transport["reduction_pct"]
t_trans.markdown(kpi([
    ("裸地参考输沙能力", f"{transport['q_bare']:.4f}", "kg/(m·s)"),
    ("植被下风侧输沙能力", f"{transport['q_vegetated']:.4f}", "kg/(m·s)"),
    ("输沙能力削减率", "—" if reduction_pct is None else f"{reduction_pct:.1f}", "%"),
    ("评价截面", f"{downstream_x:.0f}", "m"),
    ("分析高度", f"{z_probe:.1f}", "m"),
]), unsafe_allow_html=True)
if reduction_pct is None:
    t_trans.warning("裸地参考工况未达到粒径分布对应的起动条件，输沙能力为零，无法定义削减率。")
elif reduction_pct < 0:
    t_trans.info("削减率为负：该参数组合下下风侧局部输沙能力估算高于裸地参考值，建议检查适用范围。")
inter_fig = make_interception_curve(old_u, old_porosity, old_height, old_ti,
    old_leaf, old_roughness, old_moisture, particle_diameters, particle_masses, old_cover,
    downstream_x=downstream_x)
tw1, tw2 = t_trans.columns(2, gap="small")
section("覆盖度敏感性 · 下风侧输沙能力削减率", tw1)
show(inter_fig, tw1)
section("来流风速响应 · 裸地 vs 植被下风侧", tw2)
wind_fig = make_transport_wind_figure(old_u, old_cover, old_porosity, old_height, old_ti,
    old_leaf, old_roughness, old_moisture, particle_diameters, particle_masses,
    z_probe=z_probe, downstream_x=downstream_x)
show(wind_fig, tw2)
t_trans.markdown("<div class='mini-note'>采用粒径质量占比加权的 Bagnold 型经验表达，"
                 f"裸地等效 u* {transport['u_star_bare']:.3f} m/s、植被尾流等效 u* "
                 f"{transport['u_star_vegetated']:.3f} m/s。这里的「削减率」是<b>同一来流与床面条件下"
                 "输沙能力相对裸地参考值的降低比例</b>，不是被植被实际截留的沙量比例，"
                 "不含输沙沿程调整与饱和过程，经验系数未经实测标定。</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Tab 5 · 数据导出、公式与限制
# ------------------------------------------------------------
export_x = np.linspace(0, DOMAIN_END, 281)
export_rows = {"x_m": export_x, "vegetation_cover": np.full_like(export_x, old_cover),
               "base_porosity": np.full_like(export_x, old_porosity),
               "effective_porosity": np.full_like(export_x, p_eff),
               "vegetation_height_m": np.full_like(export_x, old_height),
               "leaf_development": np.full_like(export_x, old_leaf),
               "soil_moisture_pct": np.full_like(export_x, old_moisture),
               "roughness_length_m": np.full_like(export_x, old_roughness),
               "turbulence_intensity_inlet": np.full_like(export_x, old_ti),
               "turbulence_intensity_local": np.asarray(turbulence_scales(
                   export_x, z_probe, old_u, old_cover, old_porosity, old_height,
                   old_ti, old_leaf, old_roughness)[2]),
               "integration_dt_s": np.full_like(export_x, old_dt)}
for zi in [0.1, 0.2, 0.3, 0.6]:
    export_rows[f"wind_z_{zi:.1f}m_m_s"] = local_wind(export_x, zi, old_u, old_cover,
        old_porosity, old_height, old_ti, old_leaf, old_roughness)
export_df = pd.DataFrame(export_rows)
state_labels = {MOTION_PINNED: "pinned", MOTION_SALTATION: "saltation",
                MOTION_SUSPENSION: "suspension"}
particle_df = pd.DataFrame({"particle_id": np.arange(1, len(particle_diameters) + 1),
                            "diameter_um": particle_diameters,
                            "settling_velocity_m_s": settling_velocity(particle_diameters),
                            "motion_state": [state_labels[m] for m in sim["modes"]],
                            "representative_grain_mass_kg": particle_masses})
d1, d2 = t_doc.columns(2, gap="small")
d1.download_button("导出风场分析数据 (CSV)",
                   data=export_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"wind_sand_field_analysis_{APP_VERSION}.csv", mime="text/csv",
                   width="stretch")
d2.download_button("导出粒径与运动状态 (CSV)",
                   data=particle_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"wind_sand_particle_states_{APP_VERSION}.csv", mime="text/csv",
                   width="stretch")

check_exp = t_doc.expander("模型自检 · 回归测试（26 项）")
with check_exp:
    # 自检要跑 3 次短模拟，放在加载时会让首屏慢 5–6 s；改为按需触发，结果缓存
    if check_exp.button("运行自检（约 5 s）", key="run_model_checks"):
        st.session_state["check_rows"] = model_check_rows()
    check_rows = st.session_state.get("check_rows")
    if not check_rows:
        check_exp.markdown(
            "<div class='mini-note'>点击上方按钮运行；检查地表无滑移、非负性、无植被退化、"
            "覆盖度/孔隙率/高度单调性、来流响应与尾流恢复、边缘衔接平滑性、湍流分层、"
            "步长收敛、粒子数与种子波动、沉降速度与拦截方向。"
            "全量版（含 0.002 s 步长扫描）用命令行 <code>python model_checks.py</code>。</div>",
            unsafe_allow_html=True)
    else:
        n_pass = sum(1 for r in check_rows if r["_ok"])
        check_exp.markdown(
            f"<div class='mini-note'>快速档 {len(check_rows)} 项，通过 <b>{n_pass}</b> 项；"
            "全量版（含 0.002 s 步长、粒子数与多种子扫描）用 <code>python model_checks.py</code>。</div>",
            unsafe_allow_html=True)
        rows_html = "".join(
            f"<tr><td>{r['检查项']}</td>"
            f"<td class='{'ok' if r['_ok'] else 'bad'}'>{'✓ 通过' if r['_ok'] else '✗ 未通过'}</td>"
            f"<td>{r['说明']}</td></tr>" for r in check_rows)
        check_exp.markdown(
            "<table class='check-tbl'><thead><tr><th style='width:42%'>检查项</th>"
            "<th style='width:11%'>结果</th><th>实测值</th></tr></thead>"
            f"<tbody>{rows_html}</tbody></table>", unsafe_allow_html=True)

formula_exp = t_doc.expander("核心模型公式（对应当前实现）")
formula_exp.markdown("<div class='mini-note'>公式与说明文字使用同一字号与字体；"
                     "数学符号由 KaTeX 专用字体绘制。以下关系对应本程序当前实现，"
                     "经验参数与简化假设请结合「模型假设与使用限制」阅读。</div>",
                     unsafe_allow_html=True)
f1, f2 = formula_exp.columns(2)
with f1:
    f1.markdown("**01 · 中性层结对数风速廓线**")
    f1.latex(r"U(z)=\frac{u_*}{\kappa}\ln\left(\frac{z+z_0}{z_0}\right)")
    f1.latex(r"u_*=\frac{\kappa U_{10}}{\ln\left((10+z_0)/z_0\right)}")
    f1.markdown("<div class='formula-note'>κ=0.4；本实现令 U(0)=0，不解析毫米尺度黏性底层。</div>", unsafe_allow_html=True)
    f1.markdown("**02 · 植被阻力与局部风速**")
    f1.latex(r"U(x,z)=U_0(z)\left[1-0.8\left(1-e^{-D_a/27}\right)E(x,z)\right]")
    f1.markdown("<div class='formula-note'>Dₐ 为经验冠层阻力数，E(x,z) 为植被空间—垂向权重；该式是参数化标量场，不是 CFD 动量方程。</div>", unsafe_allow_html=True)
with f2:
    f2.markdown("**03 · 起动阈值与含水率修正**")
    f2.latex(r"u_{*ft}=A\sqrt{\frac{(\rho_s-\rho_a)gd}{\rho_a}}\left[1+S\left(\frac{d_c}{d}\right)^{1.5}\right]")
    f2.latex(r"u_{*ft,w}=u_{*ft}(1+4w+20w^2)")
    f2.markdown("<div class='formula-note'>Bagnold 平方根标度 + 细颗粒黏聚回升；含水率修正是教学用经验关系。</div>", unsafe_allow_html=True)
    f2.markdown("**04 · 沙粒运动状态判据（2.2 新增）**")
    f2.latex(r"u_{*e}\leq u_{*ft,w}\Rightarrow\text{静止},\quad w_s<u_*(0.8+1.6I_u)\Rightarrow\text{悬移}")
    f2.latex(r"\text{其余}\Rightarrow\text{跃移},\qquad w_s=\sqrt{\frac{4(\rho_s-\rho_a)gd}{3\rho_a C_d(Re)}}")
    f2.markdown("<div class='formula-note'>w_s 由重力—阻力平衡迭代求得；悬移限为教学用判据，不是严格 Rouse 数判据。</div>", unsafe_allow_html=True)
formula_exp.markdown("**05 · 粒径加权输沙能力（Bagnold 型估算）**")
formula_exp.latex(r"q=\sum_i\frac{m_i}{\sum_j m_j}\left[\frac{1.5\rho_a}{g}u_{*e}\max\left(u_{*e}^2-u_{*ft,w,i}^2,0\right)\right]")
formula_exp.markdown("<div class='formula-note'>单位 kg/(m·s)；是输沙能力估算，不代表实测输沙通量或植被实际截留率。</div>", unsafe_allow_html=True)
formula_exp.markdown("**06 · 逐点湍流强度（2.4 新增）**")
formula_exp.latex(r"\sigma_u=\sqrt{\sigma_b^2+\sigma_v^2},\quad \sigma_b=I_u\,U_{bare}(z),\quad I_{loc}=\frac{\sigma_u}{U_{loc}}")
formula_exp.latex(r"\sigma_v=C_t\,u_*\sqrt{c\,(1-p_{eff})}\,\Phi(x)\,\Psi(z/H),\quad \sigma_w=0.55\,\sigma_u")
formula_exp.markdown("<div class='formula-note'>Φ(x)：植被带内为 1，下风侧按尾流长度指数衰减；Ψ：冠层剪切层核，峰值约 0.8H；C_t 为经验系数。背景项按未扰动风速标定，因此冠层内 I 随局地风速下降而升高。</div>", unsafe_allow_html=True)
formula_exp.markdown("**07 · 动态状态转换与植被拦截（2.4 新增）**")
formula_exp.latex(r"w_s<u_{*e}(0.8+1.6I_{loc})\Rightarrow\text{悬移},\quad u_{*e}>u_{*ft}\Rightarrow\text{再起动}")
formula_exp.latex(r"P_{cap}=1-\exp\left[-\lambda\,c\,(1-p_{eff})\,\frac{H}{2}\,e^{-z/0.55H}\Delta t\right]")
formula_exp.markdown("<div class='formula-note'>状态每 0.016 s 按局地条件重判，颗粒进出植被区即改变状态；λ 为经验拦截速率系数，不含枝叶尺度绕流与二次溅射。</div>", unsafe_allow_html=True)

t_doc.markdown("<div class='mini-note'>15 条假设按主题分三组折叠，答辩时只展开与结论相关的一组即可；"
               "完整论证另见 README。</div>", unsafe_allow_html=True)

lim_wind = t_doc.expander("假设 ① 风场与湍流（6 条）")
lim_wind.markdown("""
1. 给出的是**参数化水平风速标量场 U(x,z)**，未求解连续性方程与动量方程，没有垂直速度分量、位移高度、冠层上方加速或回流泡；不能称为二维流场或 CFD 结果。
2. **横截面视窗是机理剖面**：冠层轮廓、风矢与亏损带按当前参数绘制，风矢长度与 local_wind 成正比，不是求解得到的流场。
3. 风场采用中性层结对数廓线，以 10 m 参考风速归一化，地表 z=0 严格为零；粗糙度长度改变廓线形状与摩阻速度，但不解析黏性底层。
4. 植被为均匀单层冠层，覆盖度/孔隙率/高度/展叶率分别控制阻力、开孔、垂向尺度与叶片附加阻力；阻力与尾流均为经验参数化。
5. **湍流是参数化强度场**：σu 由背景项（按未扰动风速标定）与植被诱导项合成，脉动由 OU 过程生成；不含真实能谱、压力—应变再分配与各向异性张量，σw/σu 固定 0.55。
6. 粗糙度遮蔽份额在 z0 ≥ 参考沙面粗糙度时随 z0 单调下降，与 u* 随 z0 上升竞争，使有效摩阻速度整体呈单峰。
""")

lim_grain = t_doc.expander("假设 ② 颗粒与运动状态（7 条）")
lim_grain.markdown("""
1. **状态判据是逐时步分类**：静止/跃移/悬移每 0.016 s 按局地等效摩阻速度与局地湍流强度重判，阈值系数（0.8、1.6）为教学取值，未解析饱和输沙层。
2. **植被拦截是概率模型**：按「覆盖度×实度×高度×叶面积垂向分布」做泊松抽样，不含枝叶尺度绕流、碰撞反弹与二次溅射。
3. 颗粒受力为重力、Schiller–Naumann 阻力与**可选** Saffman 升力（0.3 折减）；不含 Magnus 旋转、颗粒间碰撞与连锁溅射。
4. 撞地反弹系数 0.35 为经验值；落点用壁面线性插值定位，使结果基本不随步长变化。
5. 土壤含水率通过抬升临界摩阻风速抑制起动，为教学用经验关系，未针对特定土壤标定。
6. 起动阈值用 Bagnold √d 标度叠加细颗粒黏聚回升，能定性再现 U 形曲线；系数 A 与黏聚强度为教学取值。
7. 初始粒子沿上游至植被前缘分布，代表某一时刻的颗粒群快照，不是连续释放的稳态输沙；风沙为**单向耦合**。
""")

lim_scope = t_doc.expander("假设 ③ 输沙能力与适用边界（2 条）")
lim_scope.markdown("""
1. 输沙能力采用粒径质量占比加权的 Bagnold 型经验表达；界面中的「削减率」是相对于裸地参考值的降低比例，**不是被植被实际截留的沙量比例**，也不含沿程饱和调整。
2. 本程序用于课程设计原型与算法学习，不能用于工程预测或替代野外观测/风洞实验。
""")

st.caption(f"{APP_VERSION} · 物理核心 wss_model.py + 回归测试 model_checks.py：逐点湍流场、动态状态转换、植被拦截与再起动、可选 Saffman 升力；风场与输沙仍为参数化标量模型，非 CFD。")
