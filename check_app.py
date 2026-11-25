"""自动化冒烟测试：界面可运行 + 参数联动 + 模型回归。

用法：
    python check_app.py          # 界面（AppTest）快速验证
    python model_checks.py       # 仅物理模型回归（含全量扫描）

检查内容：
    1. 首屏渲染无异常（5 个标签页 / 侧栏控件 / 两个导出按钮）
    2. 缓存命中：同参数二次运行不报错
    3. 参数联动：改风速后出现「参数已改变」提示，点「运行」后提示消失且结果更新
    4. 极端工况：无植被、全静止（高含水率）、高植被 + 细粒径（悬移）均能渲染
    5. 模型自检：model_checks 全部通过
"""
import sys

from streamlit.testing.v1 import AppTest

from model_checks import collect_checks


def new_app():
    at = AppTest.from_file("main.py", default_timeout=600)
    at.run()
    return at


def fail(msg):
    print(f"[FAIL] {msg}")
    return 1


def main():
    problems = 0

    at = new_app()
    if at.exception:
        return fail(f"首屏异常：{at.exception}")
    print(f"[PASS] 首屏渲染：{len(at.tabs)} 个标签页、{len(at.sidebar.slider) + len(at.sidebar.select_slider)} 个滑块、"
          f"{len(at.download_button)} 个导出按钮")

    at.run()
    if at.exception:
        return fail(f"缓存二次运行异常：{at.exception}")
    print("[PASS] 缓存命中：同参数二次运行无异常")

    # 参数联动：改风速 -> 提示重算 -> 点运行 -> 提示消失
    at.sidebar.slider[0].set_value(14.0).run()
    if at.exception:
        return fail(f"改参数后异常：{at.exception}")
    stale = [i.value for i in at.info if "参数已改变" in i.value]
    if not stale:
        problems += 1
        print("[FAIL] 修改参数后未提示「参数已改变」")
    else:
        print("[PASS] 修改参数后提示重算")
    at.sidebar.button[0].click().run()
    if at.exception:
        return fail(f"点击运行后异常：{at.exception}")
    stale = [i.value for i in at.info if "参数已改变" in i.value]
    if stale:
        problems += 1
        print("[FAIL] 点击运行后仍提示参数已改变（结果未刷新）")
    else:
        print("[PASS] 点击运行后结果刷新、提示消失")

    # 极端工况：无植被 / 全静止 / 高植被 + 细粒径
    cases = [("无植被", [(1, 0)]),
             ("全静止（含水率 20%）", [(4, 20.0)]),
             ("高植被细粒径", [(3, 8.0), (5, 90)])]
    for label, edits in cases:
        at2 = new_app()
        for idx, value in edits:
            at2.sidebar.slider[idx].set_value(value)
        at2.run()
        if at2.exception:
            problems += 1
            print(f"[FAIL] {label} 工况异常：{at2.exception}")
            continue
        at2.sidebar.button[0].click().run()
        if at2.exception:
            problems += 1
            print(f"[FAIL] {label} 工况重算异常：{at2.exception}")
        else:
            print(f"[PASS] {label} 工况可渲染")

    # 预设：下拉选择即换工况并自动重算
    at3 = new_app()
    at3.sidebar.selectbox[0].select("稀疏灌丛").run()
    if at3.exception:
        problems += 1
        print(f"[FAIL] 预设下拉异常：{at3.exception}")
    elif at3.session_state["cover_pct"] == 20 and at3.session_state["sim_params"][1] == 0.2:
        print("[PASS] 工况预设：下拉回填并自动重算")
    else:
        problems += 1
        print("[FAIL] 预设未生效或未触发重算")
    at3.sidebar.selectbox[0].select("恢复默认参数").run()
    if at3.session_state["cover_pct"] == 30 and at3.session_state["u_ref"] == 8.0:
        print("[PASS] 恢复默认参数")
    else:
        problems += 1
        print("[FAIL] 恢复默认参数未生效")

    # 分享链接：URL 参数应作为控件初值并直接算出对应工况
    at4 = AppTest.from_file("main.py", default_timeout=600)
    at4.query_params["u"] = "12.5"
    at4.query_params["c"] = "45"
    at4.query_params["lift"] = "0"
    at4.run()
    if (not at4.exception and at4.session_state["u_ref"] == 12.5
            and at4.session_state["cover_pct"] == 45
            and at4.session_state["sim_params"][0] == 12.5):
        print("[PASS] 带参数的链接可复现工况")
    else:
        problems += 1
        print(f"[FAIL] URL 参数未正确应用：{at4.exception}")

    # 缓存正确性：缓存结果必须与「已保存参数集」对应，导出与页面同源
    from wss_model import friction_velocity
    ss = at.session_state
    sp, sim = ss["sim_params"], ss["sim_result"]
    consistent = (len(sim["modes"]) == sp[8] and sim["dt"] == sp[14]
                  and abs(sim["u_star"] - float(friction_velocity(sp[0], sp[12]))) < 1e-12)
    if consistent:
        print("[PASS] 缓存与页面/导出的参数集一致（u*、Δt、粒子数）")
    else:
        problems += 1
        print("[FAIL] 缓存结果与当前参数集不一致")

    # 界面里的自检按钮：点击后应产出表格且不报错
    btns = [b for b in at.button if b.label.startswith("运行自检")]
    if btns:
        btns[0].click().run()
        if at.exception or not at.session_state.get("check_rows"):
            problems += 1
            print(f"[FAIL] 界面自检按钮异常：{at.exception}")
        else:
            print("[PASS] 界面自检按钮可运行")
    else:
        problems += 1
        print("[FAIL] 未找到界面自检按钮")

    rows = collect_checks(fast=True)
    bad = [r for r in rows if not r["_ok"]]
    if bad:
        problems += len(bad)
        for r in bad:
            print(f"[FAIL] 模型自检：{r['检查项']} — {r['说明']}")
    else:
        print(f"[PASS] 模型自检 {len(rows)}/{len(rows)} 项通过")

    print("\n全部通过" if problems == 0 else f"\n{problems} 项未通过")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
