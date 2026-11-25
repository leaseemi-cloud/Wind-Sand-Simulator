"""模型回归测试：不启动 Streamlit，直接检查 wss_model 的物理不变量与数值行为。

用法：
    python model_checks.py            # 全量检查（含步长/粒子数/种子扫描，约 20 s）
    python model_checks.py --fast     # 仅快速检查（约 2 s，界面「模型自检」用）

检查项覆盖：
    1. 风场：地表无滑移、非负、无植被退化、覆盖度/孔隙率/高度单调性、
       来流风速响应、尾流恢复距离、植被边缘与尾流衔接平滑性
    2. 湍流：冠层 > 尾流 > 来流、无植被退化为常量、植被脉动峰值位置
    3. 数值：步长收敛、粒子数无关性、种子波动、状态守恒、轨迹有限性
    4. 颗粒：沉降速度量级与单调性、起动阈值、输沙能力削减方向、植被拦截
"""
from __future__ import annotations

import sys

import numpy as np

import wss_model as M

U0, C0, P0, H0, TI0 = 8.0, 0.30, 0.40, 2.0, 0.15
LEAF0, Z00, MO0 = 0.75, 0.0005, 2.0


def _center_reduction(u_ref=U0, cover=C0, porosity=P0, height=H0, z=0.2, **kw):
    bare = float(M.wind_profile(z, u_ref, Z00)[0])
    veg = float(M.local_wind((M.VEG_START + M.VEG_END) / 2, z, u_ref, cover,
                             porosity, height, kw.get("ti", TI0), LEAF0, Z00))
    return (1.0 - veg / bare) * 100.0


def _sim(**kw):
    args = dict(u_ref=U0, cover=C0, mean_diameter_um=200.0, n_particles=200,
                porosity=P0, veg_height=H0, turbulence_intensity=TI0, seed=7,
                soil_moisture_pct=MO0, leaf_development=LEAF0, roughness=Z00)
    args.update(kw)
    return M.simulate_saltation(
        args.pop("u_ref"), args.pop("cover"), args.pop("mean_diameter_um"),
        args.pop("n_particles"), **args)


def collect_checks(fast: bool = True) -> list[dict]:
    rows: list[dict] = []
    notes: dict[str, str] = {}

    def check(name, ok, detail=""):
        rows.append({"检查项": name, "结果": "通过" if ok else "未通过",
                     "说明": detail, "_ok": bool(ok)})

    # ---------------- 1. 风场 ----------------
    worst_zero = 0.0
    for u in (2.0, 8.0, 20.0):
        for z0 in (1e-4, 5e-4, 1e-2):
            worst_zero = max(worst_zero, abs(float(M.wind_profile(0.0, u, z0)[0])))
            for c in (0.0, 0.3, 0.8):
                worst_zero = max(worst_zero, abs(float(M.local_wind(
                    45.0, 0.0, u, c, P0, H0, TI0, LEAF0, z0))))
    check("地表无滑移 U(z=0)=0（18 组参数）", worst_zero <= 1e-12,
          f"最大残差 {worst_zero:.1e} m/s")

    xx = np.linspace(0.0, M.DOMAIN_END, 141)
    zz = np.linspace(0.0, 8.0, 81)
    grid_min = 1e9
    for u in (2.0, 8.0, 20.0):
        for c, p, h in ((0.0, 0.4, 2.0), (0.3, 0.0, 6.0), (0.8, 0.8, 0.5)):
            val = M.local_wind(xx[:, None], zz[None, :], u, c, p, h, 0.40, 0.0, 1e-4)
            grid_min = min(grid_min, float(np.min(val)))
    check("风速场非负（3 风速 × 3 极端工况）", grid_min >= 0.0,
          f"最小 U = {grid_min:.3f} m/s")

    deg = 0.0
    for z0 in (1e-4, 5e-4, 1e-2):
        for z in (0.05, 0.2, 1.0, 4.0):
            bare = float(M.wind_profile(z, U0, z0)[0])
            veg = float(M.local_wind(45.0, z, U0, 0.0, P0, H0, TI0, LEAF0, z0))
            deg = max(deg, abs(bare - veg))
        deg = max(deg, abs(float(M.local_friction_velocity(
            45.0, U0, 0.0, P0, H0, TI0, LEAF0, z0)) - float(M.friction_velocity(U0, z0))))
    check("无植被严格退化为裸地风场", deg <= 1e-9, f"最大偏差 {deg:.1e}")

    cover_series = [_center_reduction(cover=c) for c in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8)]
    check("覆盖度 ↑ → 中心降低率单调 ↑", np.all(np.diff(cover_series) > 0),
          " ".join(f"{v:.1f}%" for v in cover_series))

    poro_series = [_center_reduction(porosity=p) for p in (0.0, 0.2, 0.4, 0.6, 0.8)]
    check("孔隙率 ↑ → 中心降低率单调 ↓", np.all(np.diff(poro_series) < 0),
          " ".join(f"{v:.1f}%" for v in poro_series))

    height_series = [_center_reduction(height=h) for h in (0.5, 1.0, 2.0, 4.0, 6.0)]
    check("植被高度 ↑ → 中心降低率单调 ↑", np.all(np.diff(height_series) > 0),
          " ".join(f"{v:.1f}%" for v in height_series))

    u_list = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    red_u = np.array([_center_reduction(u_ref=u) for u in u_list])
    check("来流风速 ↑ → 中心降低率单调不减且饱和",
          np.all(np.diff(red_u) >= -1e-9) and (red_u.max() - red_u.min()) <= 10.0,
          f"{red_u.min():.1f}%–{red_u.max():.1f}%（跨度 {red_u.max()-red_u.min():.1f} pp）")

    rec_u = [M.recovery_distance(u, C0, P0, H0, turbulence_intensity=TI0,
                                 leaf_development=LEAF0, roughness=Z00) for u in u_list]
    check("来流风速 ↑ → 尾流恢复距离单调 ↑",
          all(r is not None for r in rec_u) and np.all(np.diff([float(r) for r in rec_u]) > 0),
          " ".join(f"{float(r):.0f}" for r in rec_u) + " m")

    x_line = np.linspace(0.0, M.DOMAIN_END, 8001)
    u_line = M.local_wind(x_line, 0.2, U0, C0, P0, H0, TI0, LEAF0, Z00)
    grad = float(np.max(np.abs(np.diff(u_line))) / (x_line[1] - x_line[0]))
    jump = max(abs(float(M.local_wind(24.99, 0.2, U0, C0, P0, H0, TI0, LEAF0, Z00))
                   - float(M.local_wind(25.01, 0.2, U0, C0, P0, H0, TI0, LEAF0, Z00))),
               abs(float(M.local_wind(64.99, 0.2, U0, C0, P0, H0, TI0, LEAF0, Z00))
                   - float(M.local_wind(65.01, 0.2, U0, C0, P0, H0, TI0, LEAF0, Z00))))
    u_bare_probe = float(M.wind_profile(0.2, U0, Z00)[0])
    grad_rel = grad / u_bare_probe
    jump_rel = jump / u_bare_probe
    check("植被前缘 / 尾流衔接平滑（梯度与阶跃均 <5%/m、1%）",
          grad_rel <= 0.05 and jump_rel <= 0.01,
          f"梯度 {grad_rel:.2%}/m，边缘跳变 {jump_rel:.2%}")

    # ---------------- 2. 湍流 ----------------
    i_can = float(np.asarray(M.turbulence_scales(45.0, 0.2, U0, C0, P0, H0, TI0, LEAF0, Z00)[2]))
    i_wake = float(np.asarray(M.turbulence_scales(80.0, 0.2, U0, C0, P0, H0, TI0, LEAF0, Z00)[2]))
    i_free = float(np.asarray(M.turbulence_scales(10.0, 0.2, U0, C0, P0, H0, TI0, LEAF0, Z00)[2]))
    check("湍流强度：冠层 > 尾流 > 来流", i_can > i_wake > i_free >= TI0 - 1e-9,
          f"冠层 {i_can:.0%} / 尾流 {i_wake:.0%} / 来流 {i_free:.0%}")

    i_bare = float(np.asarray(M.turbulence_scales(45.0, 0.2, U0, 0.0, P0, H0, TI0, LEAF0, Z00)[2]))
    check("无植被时湍流强度等于来流设定值", abs(i_bare - TI0) <= 1e-9, f"{i_bare:.0%}")

    z_prof = np.linspace(0.01, 2.0 * H0, 400)
    sig_v = M.vegetation_turbulence(np.full_like(z_prof, 45.0), z_prof, U0, C0, P0, H0,
                                    TI0, LEAF0, Z00)
    z_peak = float(z_prof[int(np.argmax(sig_v))])
    check("植被脉动在冠层上部（0.5–1.0 H）取峰值",
          0.5 * H0 <= z_peak <= 1.0 * H0, f"峰值高度 {z_peak:.2f} m（H={H0:.1f} m）")

    # ---------------- 3. 颗粒与数值 ----------------
    base = _sim(n_particles=400)
    counts = base["state_counts"]
    total = sum(counts.values()) + base["intercepted"]
    check("状态计数守恒（静止+跃移+悬移+拦截=粒子数）", total == 400,
          f"{counts['pinned']}+{counts['saltation']}+{counts['suspension']}"
          f"+{base['intercepted']}={total}")

    all_xy = [(px, pz) for px, pz, *_ in base["trajectories"]]
    finite = all(np.all(np.isfinite(px)) and np.all(np.isfinite(pz)) for px, pz in all_xy)
    in_domain = (np.all([np.min(px) >= -0.05 for px, _ in all_xy])
                 and np.all([np.max(px) <= M.DOMAIN_END + 1.0 for px, _ in all_xy]))
    check("轨迹坐标有限且位于计算域内", finite and in_domain,
          f"{len(all_xy)} 条轨迹")

    zmax = max((max(pz) for _, pz in all_xy), default=0.0)
    check("跃移层高度处于物理量级（< 1 m）", zmax < 1.0, f"最高点 {zmax:.2f} m")

    d_test = np.array([80.0, 150.0, 200.0, 320.0, 500.0])
    ws = M.settling_velocity(d_test)
    check("沉降速度量级与粒径单调性（200 μm 约 1.0–1.7 m/s）",
          np.all(np.diff(ws) > 0) and 1.0 <= float(ws[2]) <= 1.7,
          " ".join(f"{v:.2f}" for v in ws) + " m/s")

    th = M.fluid_threshold(np.array([80.0, 150.0, 250.0, 400.0]))
    check("起动阈值随粗颗粒粒径单调增、细颗粒受黏聚抬升",
          np.all(np.diff(th[1:]) > 0) and float(th[0]) > 0.9 * float(th[1]),
          " ".join(f"{v:.3f}" for v in th) + " m/s")

    q_bare = M.transport_capacity_bagnold(0.32, base["diameters_um"], base["masses"], MO0, Z00)
    q_veg = M.transport_capacity_bagnold(0.26, base["diameters_um"], base["masses"], MO0, Z00)
    check("植被削弱风速后输沙能力下降", q_veg < q_bare,
          f"{q_bare:.4f} → {q_veg:.4f} kg/(m·s)")

    if fast:
        # 快速档只做两次步长，避免界面首次加载等待过久
        dt_runs = [(0.008, 200), (0.004, 200)]
    else:
        dt_runs = [(0.008, 200), (0.004, 200), (0.002, 200)]

    hops, valid = [], True
    for dt_val, n_p in dt_runs:
        r = _sim(dt=dt_val, n_particles=n_p)
        h = np.asarray(r["hop_lengths"], dtype=float)
        valid = valid and h.size > 0 and np.all(np.isfinite(h))
        hops.append(float(np.median(h)) if h.size else np.nan)
    rel = abs(hops[0] - hops[-1]) / max(hops[-1], 1e-9)
    check("积分步长收敛（0.008 vs 0.004/0.002 s，跃移距离差 <15%）",
          valid and rel < 0.15, " / ".join(f"{v:.3f}" for v in hops) + f" m（差 {rel:.1%}）")

    if not fast:
        frac = []
        for n_p in (100, 600):
            r = _sim(n_particles=n_p)
            c = r["state_counts"]
            frac.append(c["saltation"] / max(sum(c.values()) + r["intercepted"], 1))
        check("粒子数 100 → 600 的跃移比例差 <15 pp",
              abs(frac[0] - frac[1]) < 0.15,
              f"{frac[0]:.0%} vs {frac[1]:.0%}")

        seeds = []
        for sd in (1, 2, 3, 4, 6):
            r = _sim(seed=sd, n_particles=300)
            seeds.append(r["state_counts"]["saltation"]
                         / max(sum(r["state_counts"].values()) + r["intercepted"], 1))
        cv = float(np.std(seeds, ddof=1) / max(np.mean(seeds), 1e-9))
        check("随机种子波动（跃移比例 CV < 20%）", cv < 0.20,
              f"CV {cv:.1%}（{min(seeds):.0%}–{max(seeds):.0%}）")

        inter = []
        for cv_val in (0.2, 0.6):
            r = _sim(cover=cv_val, n_particles=300)
            inter.append(r["intercepted"] / 300.0)
        check("覆盖度 ↑ → 植被拦截比例 ↑", inter[1] > inter[0],
              f"{inter[0]:.0%} vs {inter[1]:.0%}")

        # 湍流强度对颗粒轨迹的影响：强湍流把细颗粒抬升为悬移
        susp_frac = []
        for ti_val in (0.05, 0.40):
            r = _sim(u_ref=12.0, turbulence_intensity=ti_val, mean_diameter_um=100.0,
                     n_particles=250)
            susp_frac.append(r["state_counts"]["suspension"] / 250.0)
        check("湍流强度 5%→40% → 悬移比例显著上升",
              susp_frac[1] - susp_frac[0] >= 0.10,
              f"{susp_frac[0]:.0%} → {susp_frac[1]:.0%}")

        # 粒径差异：粗颗粒更难悬移、单次跃移更远
        coarse, fine = (_sim(u_ref=12.0, mean_diameter_um=d, n_particles=250)
                        for d in (300.0, 100.0))
        hop_f = float(np.median(fine["hop_lengths"])) if fine["hop_lengths"] else 0.0
        hop_c = float(np.median(coarse["hop_lengths"])) if coarse["hop_lengths"] else 0.0
        check("粒径 100→300 μm：悬移减少、跃移距离增大",
              coarse["state_counts"]["suspension"] < fine["state_counts"]["suspension"]
              and hop_c > hop_f,
              f"悬移 {fine['state_counts']['suspension']}→{coarse['state_counts']['suspension']}，"
              f"跃移 {hop_f:.2f}→{hop_c:.2f} m")

        # 孔隙率与高度：疏透度越低、冠层越高，拦截越强
        poro_inter = [(_sim(porosity=pv, n_particles=250)["intercepted"] / 250.0)
                      for pv in (0.15, 0.70)]
        check("孔隙率 15%→70% → 拦截比例下降", poro_inter[0] > poro_inter[1],
              f"{poro_inter[0]:.0%} vs {poro_inter[1]:.0%}")
        height_inter = [(_sim(veg_height=hv, n_particles=250)["intercepted"] / 250.0)
                        for hv in (0.5, 4.0)]
        check("植被高度 0.5→4 m → 拦截比例上升", height_inter[1] > height_inter[0],
              f"{height_inter[0]:.0%} vs {height_inter[1]:.0%}")

    return rows


def main(fast: bool = False) -> int:
    rows = collect_checks(fast=fast)
    width = max(len(r["检查项"]) for r in rows)
    print(f"Wind & Sand Simulator {M.APP_VERSION} · 模型回归检查"
          f"（{'快速' if fast else '全量'}）\n")
    for r in rows:
        flag = "PASS" if r["_ok"] else "FAIL"
        print(f"[{flag}] {r['检查项']:<{width}}  {r['说明']}")
    failed = [r for r in rows if not r["_ok"]]
    print(f"\n{len(rows) - len(failed)}/{len(rows)} 项通过")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main(fast="--fast" in sys.argv))
