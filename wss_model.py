"""Wind & Sand Simulator · 物理核心（纯计算，无界面依赖）。

本模块只包含常数、风场 / 湍流 / 起动阈值 / 颗粒动力学 / 输沙能力，
可在不启动 Streamlit 的情况下导入，供 model_checks.py 做回归测试。
"""
import numpy as np
import streamlit as st

APP_VERSION = "2.4"

# 缓存只在 Streamlit 运行时内生效；命令行导入（回归测试）时退化为普通函数，
# 既不打印 "No runtime found" 警告，也保证测试拿到的一定是新算的结果。
if st.runtime.exists():          # pragma: no cover - 分支取决于运行环境
    cache_data = st.cache_data
else:
    def cache_data(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap

KAPPA = 0.4
RHO_AIR = 1.225
RHO_SAND = 2650.0
G = 9.81
MU_AIR = 1.81e-5
Z_REF = 10.0
VEG_START = 25.0
VEG_END = 65.0
DOMAIN_END = 140.0

DEFAULT_VEG_HEIGHT = 2.0
DEFAULT_POROSITY = 0.40
DEFAULT_ROUGHNESS = 0.0005
DEFAULT_LEAF_DEVELOPMENT = 0.75

Z_EVAL = 0.05              # 跃移层代表高度，用于反演局部等效摩阻速度
THRESHOLD_A = 0.0872       # Bagnold 系数 A，标定至 d=250 um 时阈值约 0.22 m/s
D_COHESION = 100e-6        # 细颗粒黏聚作用特征粒径
COHESION_STRENGTH = 0.35   # 黏聚导致的阈值回升强度
SUSTAIN_RATIO = 0.75       # 冲击(维持)阈值 / 流体起动阈值
T_LAGRANGIAN = 0.30        # 拉格朗日积分时间尺度 (s)
DT = 0.004                 # 默认积分步长 (s)
DT_OPTIONS = (0.002, 0.004, 0.008)   # 步长敏感性可选值
OBS_WINDOW = 20.0          # 观察窗长度 (s)，步长改变时保持窗口不变
SNAPSHOT_DT = 0.064        # 轨迹快照间隔 (s)
STATE_UPDATE_EVERY = 4     # 局地风速/状态判据每 N 步更新一次（缓变量，省 3/4 开销）

# 2.2：逐颗粒运动状态
MOTION_PINNED = 0          # 静止：有效摩阻速度未超过含水率/黏聚修正后的起动阈值
MOTION_SALTATION = 1       # 跃移：溅射维持的弹跳
MOTION_SUSPENSION = 2      # 悬移：沉降速度小于湍流增强后的悬移限
SUSPENSION_BASE = 0.80     # 悬移判据 w_s < u* (base + turb * Iu)
SUSPENSION_TURB = 1.60     # 湍流对悬移判据的增强系数

# 2.4：湍流模型
TURB_VEG_COEFF = 3.5       # 植被诱导脉动强度 / u*（冠层剪切层峰值）
TURB_WAKE_DECAY = 0.85     # 植被带末端处诱导湍流保留比例
SIGMA_W_RATIO = 0.55       # σw / σu（地表层观测约 0.5，冠层内略高）
CANOPY_SHEAR_Z = 0.85      # 最大剪切生成高度 / 冠层高度

# 2.4：颗粒—植被相互作用与动力学
INTERCEPT_RATE = 0.06      # 拦截速率系数 (1/s)，按覆盖度×实度×高度缩放
REENTRAIN_RATE = 1.0       # 静止颗粒再起动速率 (1/s)
LIFT_COEFF = 0.30          # Saffman 升力折减系数（抑制床面附近的奇异放大）
MIN_DYNAMIC_Z = 0.0025     # 计算风速切变时的最小高度 (m)

STATE_COLORS = {MOTION_PINNED: "#64748B", MOTION_SALTATION: "#E0A83B",
                MOTION_SUSPENSION: "#3B82F6"}
DEPOSIT_COLOR = "#E05C4B"


# ------------------------------------------------------------
# 1. 基准风廓线与摩阻速度
# ------------------------------------------------------------

def friction_velocity(u_ref, roughness=DEFAULT_ROUGHNESS):
    """由 10 m 参考风速与粗糙度长度反演摩阻速度 u*。

    全局唯一的 u* 闭合，避免各子模块各自计算导致粗糙度传参不一致。
    """
    z0 = float(np.clip(roughness, 1e-5, 0.05))
    u = max(float(u_ref), 0.1)
    return KAPPA * u / np.log((Z_REF + z0) / z0)


def wind_profile(z, u_ref, roughness=DEFAULT_ROUGHNESS):
    """中性层结对数廓线；以 10 m 参考风速归一化，并严格满足 U(0)=0。

    roughness 为空气动力学粗糙度长度 z0。该表达是地表层经验廓线，
    不解析毫米尺度黏性底层；粗糙度同时影响廓线形状及摩阻速度。
    """
    z_arr = np.maximum(np.asarray(z, dtype=float), 0.0)
    z0 = float(np.clip(roughness, 1e-5, 0.05))
    u_star = friction_velocity(u_ref, z0)
    speed = (u_star / KAPPA) * np.log((z_arr + z0) / z0)
    return np.where(z_arr == 0.0, 0.0, speed), u_star


def effective_porosity(porosity, leaf_development):
    """叶片展开发育使冠层迎风开孔率下降；leaf_development 取 0~1。

    基础孔隙率表示落叶/未展叶状态下枝干结构的开孔率。
    完全展叶时，叶片使有效开孔率最多再降低 55%。
    """
    p = float(np.clip(porosity, 0.0, 0.95))
    leaf = float(np.clip(leaf_development, 0.0, 1.0))
    return float(np.clip(p * (1.0 - 0.55 * leaf), 0.01, 0.99))


# ------------------------------------------------------------
# 2. 冠层阻力、尾流与水平风速场
# ------------------------------------------------------------

def canopy_drag_number(u_ref, cover, porosity, veg_height, turbulence_intensity,
                       leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                       roughness=DEFAULT_ROUGHNESS):
    """植被阻力-混合竞争参数（无量纲）；采用二次阻力及经验混合闭合。

    混合项只由湍流强度驱动，避免与迎风面积指数中的 solidity 相互抵消，
    从而使孔隙率对冠层减速具有可辨识的单调影响。
    """
    c = float(np.clip(cover, 0.0, 1.0))
    p_eff = effective_porosity(porosity, leaf_development)
    h = float(np.clip(veg_height, 0.1, 20.0))
    ti = float(np.clip(turbulence_intensity, 0.01, 0.60))
    if c <= 0.0:
        return 0.0
    solidity = 1.0 - p_eff
    u_star = friction_velocity(u_ref, roughness)
    u = max(float(u_ref), 0.1)
    # 经验风速标定项（不是雷诺数）：冠层阻力系数随风速轻微上升
    cd_canopy = 0.55 + 0.25 * (1.0 - np.exp(-u / 5.0))
    frontal_area_index = 8.0 * c * solidity * (h / 0.8) ** 0.35
    mixing_factor = 1.0 + 7.0 * ti
    canopy_length = VEG_END - VEG_START
    return (0.5 * cd_canopy * frontal_area_index * (canopy_length / 40.0) * u
            / max(u_star * mixing_factor, 0.05))


def wake_length_scale(u_ref, cover, porosity, veg_height, turbulence_intensity=0.15,
                      leaf_development=DEFAULT_LEAF_DEVELOPMENT):
    """经验尾流特征长度 Lw（m）。

    疏透度采用单峰响应：过密冠层形成短而强的回流区，过疏冠层几乎不产生
    动量亏损，中等疏透度对应最长的有效防护距离。
    """
    c = float(np.clip(cover, 0.0, 1.0))
    p_eff = effective_porosity(porosity, leaf_development)
    h = float(np.clip(veg_height, 0.1, 20.0))
    ti = float(np.clip(turbulence_intensity, 0.01, 0.60))
    u = max(float(u_ref), 0.1)
    base = 7.0 + 11.0 * np.sqrt(c) * (u / 10.0) ** 0.25
    height_factor = (h / DEFAULT_VEG_HEIGHT) ** 0.55
    porosity_factor = 0.35 + 1.65 * np.exp(-((p_eff - 0.35) / 0.28) ** 2)
    mixing_factor = 1.0 + 2.2 * (ti - 0.15)
    return base * height_factor * porosity_factor / max(mixing_factor, 0.25)


def vegetation_effect(x, z, cover, u_ref, porosity, veg_height, turbulence_intensity=0.15,
                      leaf_development=DEFAULT_LEAF_DEVELOPMENT):
    """均匀单层植被带及其尾流的空间-垂向权重（0~1）。

    覆盖度不再在这里相乘：它已通过 canopy_drag_number 进入阻力响应，
    重复相乘会造成覆盖度被平方削弱、减速幅度系统性偏低。
    """
    x_arr, z_arr = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
    c = float(np.clip(cover, 0.0, 1.0))
    h = max(float(veg_height), 0.1)
    leading = 1.0 / (1.0 + np.exp(-(x_arr - VEG_START) / 2.5))
    inside = np.clip((VEG_END - x_arr) / 8.0, 0.0, 1.0)
    lw = wake_length_scale(u_ref, c, porosity, h, turbulence_intensity, leaf_development)
    wake = np.exp(-np.maximum(x_arr - VEG_END, 0.0) / lw)
    spatial = leading * np.where(x_arr <= VEG_END, 0.9 + 0.1 * inside, 0.9 * wake)
    z_pos = np.maximum(z_arr, 0.0)
    vertical_inside = np.exp(-z_pos / (0.70 * h))
    value_at_top = np.exp(-1.0 / 0.70)
    vertical_above = value_at_top * np.exp(-np.maximum(z_pos - h, 0.0) / (1.6 * h))
    vertical = np.where(z_pos <= h, vertical_inside, vertical_above)
    return np.clip(spatial * vertical, 0.0, 1.0)


def local_wind(x, z, u_ref, vegetation_cover, porosity=DEFAULT_POROSITY,
               veg_height=DEFAULT_VEG_HEIGHT, turbulence_intensity=0.15,
               leaf_development=DEFAULT_LEAF_DEVELOPMENT, roughness=DEFAULT_ROUGHNESS):
    """参数化水平风速；粗糙度改变基准廓线，物候改变冠层有效孔隙率。"""
    bare_u, _ = wind_profile(z, u_ref, roughness)
    effect = vegetation_effect(x, z, vegetation_cover, u_ref, porosity, veg_height,
                               turbulence_intensity, leaf_development)
    da = canopy_drag_number(u_ref, vegetation_cover, porosity, veg_height,
                            turbulence_intensity, leaf_development, roughness)
    # 指数饱和响应：低覆盖度近似线性，高覆盖度渐近上限，避免覆盖度被平方削弱
    drag_response = 1.0 - np.exp(-da / 27.0) if da > 0 else 0.0
    reduction = 0.80 * drag_response * effect
    return np.maximum(bare_u * (1.0 - reduction), 0.0)


def local_friction_velocity(x, u_ref, cover, porosity, veg_height, turbulence_intensity,
                            leaf_development, roughness, z_eval=Z_EVAL):
    """在跃移层代表高度反演局部「等效摩阻速度」。

    用与 wind_profile 相同的裸地对数律反演，因此零覆盖度时严格回到 u*。
    注意它是水平风速的等效换算量，不是冠层内真实雷诺应力或独立求解的剪切。
    """
    z0 = float(np.clip(roughness, 1e-5, 0.05))
    u = local_wind(x, z_eval, u_ref, cover, porosity, veg_height,
                   turbulence_intensity, leaf_development, roughness)
    log_term = np.log((z_eval + z0) / z0)
    return KAPPA * np.asarray(u, dtype=float) / max(log_term, 1e-9)


# ------------------------------------------------------------
# 2b. 湍流强度场：背景湍流 + 植被诱导湍流
# ------------------------------------------------------------

def canopy_shear_kernel(z, veg_height):
    """冠层湍流生成的垂向核函数：峰值在冠层上部剪切层，冠层内为常数级，冠上迅速衰减。

    canopy_shear_kernel ≈ 1 于 z≈0.85H，衰减尺度与冠层高度成正比；
    另加一项指数尾，表示冠层内部枝叶尾流造成的体源湍流。
    """
    h = max(float(veg_height), 0.1)
    z_arr = np.maximum(np.asarray(z, dtype=float), 0.0)
    shear = np.exp(-((z_arr - CANOPY_SHEAR_Z * h) / (0.55 * h + 0.30)) ** 2)
    body = 0.45 * np.exp(-z_arr / (0.70 * h))
    return shear + body


def vegetation_turbulence(x, z, u_ref, cover, porosity, veg_height,
                          turbulence_intensity=0.15,
                          leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                          roughness=DEFAULT_ROUGHNESS):
    """植被诱导的脉动速度标准差 σ_veg (m/s)。

    量级 ∝ u*·sqrt(覆盖度×实度)：实度（1−有效孔隙率）越高、覆盖度越大，
    枝叶尾流与冠层剪切越强；沿程在植被带内为满值，进入下风侧按尾流长度
    指数衰减，因此「植被区湍流强、尾流区逐渐回落」是模型的内生结果。
    """
    x_arr = np.asarray(x, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    c = float(np.clip(cover, 0.0, 1.0))
    if c <= 0.0:
        return np.zeros(np.broadcast(x_arr, z_arr).shape)
    p_eff = effective_porosity(porosity, leaf_development)
    solidity = 1.0 - p_eff
    u_star = friction_velocity(u_ref, roughness)
    lead = 1.0 / (1.0 + np.exp(-(x_arr - VEG_START) / 2.5))
    lw = wake_length_scale(u_ref, c, porosity, veg_height, turbulence_intensity,
                           leaf_development)
    stream = np.where(x_arr <= VEG_END, 1.0,
                      TURB_WAKE_DECAY * np.exp(-np.maximum(x_arr - VEG_END, 0.0) / lw))
    amplitude = TURB_VEG_COEFF * u_star * np.sqrt(c * solidity)
    return amplitude * lead * stream * canopy_shear_kernel(z_arr, veg_height)


def turbulence_scales(x, z, u_ref, cover, porosity, veg_height, turbulence_intensity=0.15,
                      leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                      roughness=DEFAULT_ROUGHNESS, u_local=None):
    """逐点湍流脉动标准差与局地湍流强度。

    返回 (σu, σw, I_local)：
    - 背景项 σb = Iu·U_local，随局地平均风速缩放（原实现用 Iu·U₁₀ 全场常数，
      导致冠层内风速已降到 1/3、脉动却仍是来流值，湍流强度被系统性低估）；
    - 植被项 σveg 由冠层剪切层 + 枝叶尾流给出，见 vegetation_turbulence；
    - σu = sqrt(σb² + σveg²)，σw = 0.55σu（地表层 σw/σu 观测约 0.5）；
    - I_local = σu / U_local，上限 90%，避免近床面 U→0 时的除零放大。
    """
    if u_local is None:
        u_local = local_wind(x, z, u_ref, cover, porosity, veg_height,
                             turbulence_intensity, leaf_development, roughness)
    u_local = np.asarray(u_local, dtype=float)
    bare, _ = wind_profile(z, u_ref, roughness)
    ti = float(np.clip(turbulence_intensity, 0.01, 0.60))
    # 背景项按「未扰动风速」标定：上游地表层湍流被平流带入冠层，
    # 其脉动能量不会随局地平均风速同比衰减（否则冠层内 I 恒等于来流值）。
    sigma_b = ti * np.maximum(np.asarray(bare, dtype=float), 0.0)
    sigma_v = vegetation_turbulence(x, z, u_ref, cover, porosity, veg_height,
                                    turbulence_intensity, leaf_development, roughness)
    sigma_u = np.sqrt(sigma_b ** 2 + np.asarray(sigma_v, dtype=float) ** 2)
    sigma_w = SIGMA_W_RATIO * sigma_u
    i_local = np.clip(sigma_u / np.maximum(u_local, 0.05), 0.0, 0.90)
    return sigma_u, sigma_w, i_local


# ------------------------------------------------------------
# 3. 起动阈值、含水率与粗糙度遮蔽
# ------------------------------------------------------------

def fluid_threshold(diameter_um):
    """流体起动摩阻风速 u*_ft (m/s)：Bagnold 平方根标度 + 细颗粒黏聚回升。

    大颗粒：u*_ft = A·sqrt((rho_s-rho_a)·g·d/rho_a)，即随 sqrt(d) 增长；
    细颗粒：粒间黏聚力占优，阈值随粒径减小而重新抬升，形成 U 形曲线。
    """
    d = np.maximum(np.asarray(diameter_um, dtype=float) * 1e-6, 1e-6)
    grain = THRESHOLD_A * np.sqrt((RHO_SAND - RHO_AIR) * G * d / RHO_AIR)
    cohesion = 1.0 + COHESION_STRENGTH * (D_COHESION / d) ** 1.5
    return grain * cohesion


def moisture_threshold(base_threshold, soil_moisture_pct):
    """土壤含水率提高颗粒黏结，抬升临界摩阻风速（教学用经验关系）。"""
    w = float(np.clip(soil_moisture_pct, 0.0, 25.0)) / 100.0
    return np.asarray(base_threshold, dtype=float) * (1.0 + 4.0 * w + 20.0 * w * w)


def roughness_partition(roughness):
    """粗糙元遮蔽导致的可用于可蚀沙面的剪切应力份额。

    以平滑沙面 z0s 为基准，采用幂律衰减并保留 0.15 的渐近下限（不再硬截断），
    保证在 z0 >= z0s 时「粗糙度越大越不易起沙」严格单调成立。
    z0 < z0s 时不再额外增额，此时 u* 随 z0 上升占主导，整体呈单峰响应：
    这对应「整体剪切增大」与「粗糙元分担应力」两种机制的竞争，是物理上合理的。
    """
    z0 = float(np.clip(roughness, 1e-5, 0.05))
    ratio = max(z0 / DEFAULT_ROUGHNESS, 1.0)
    return float(np.clip(0.15 + 0.85 * ratio ** (-0.35), 0.15, 1.0))


def validate_model_invariants():
    """启动时验证无滑移边界、粗糙度遮蔽单调性、零覆盖度退化性。"""
    test_speeds = [2.0, 8.0, 20.0]
    test_covers = [0.0, 0.2, 0.6, 0.8]
    test_porosity = [0.0, 0.2, 0.4, 0.8]
    test_heights = [0.5, 2.0, 6.0]
    test_roughness = [0.0001, 0.0005, 0.01]
    for u_ref_test in test_speeds:
        for roughness_test in test_roughness:
            bare, _ = wind_profile(0.0, u_ref_test, roughness_test)
            if not np.allclose(bare, 0.0, atol=1e-12):
                raise AssertionError("wind_profile 未满足 U(z=0)=0")
            for cover_test in test_covers:
                for porosity_test in test_porosity:
                    for height_test in test_heights:
                        value = local_wind(45.0, 0.0, u_ref_test, cover_test,
                                           porosity_test, height_test, 0.15,
                                           DEFAULT_LEAF_DEVELOPMENT, roughness_test)
                        if not np.allclose(value, 0.0, atol=1e-12):
                            raise AssertionError("local_wind 未满足 U(x,z=0)=0")
    # 仅在 z0 >= 参考沙面粗糙度区间要求严格单调；更光滑段允许 u* 上升占主导
    z0_grid = np.linspace(DEFAULT_ROUGHNESS, 0.02, 60)
    effective = np.array([friction_velocity(8.0, z) * roughness_partition(z) for z in z0_grid])
    if np.any(np.diff(effective) > 1e-12):
        raise AssertionError("z0>=z0s 区间粗糙度遮蔽应使有效摩阻速度单调下降")
    for z0_test in z0_grid[::10]:
        bare, u_star_bare = wind_profile(0.2, 8.0, z0_test)
        plain = local_wind(45.0, 0.2, 8.0, 0.0, 0.4, 2.0, 0.15, 0.75, z0_test)
        if not np.allclose(plain, bare, atol=1e-12):
            raise AssertionError("零覆盖度应严格退化为裸地廓线")
        local_u_star = local_friction_velocity(45.0, 8.0, 0.0, 0.4, 2.0, 0.15, 0.75, z0_test)
        if not np.allclose(local_u_star, u_star_bare, atol=1e-9):
            raise AssertionError("零覆盖度时局部等效摩阻速度应等于裸地摩阻速度")


validate_model_invariants()


# ------------------------------------------------------------
# 4. 颗粒阻力与粒径抽样
# ------------------------------------------------------------

def drag_coefficient(reynolds):
    """Schiller-Naumann 球面阻力系数，支持标量或数组；Re=1000 处连续。"""
    re = np.maximum(np.asarray(reynolds, dtype=float), 1e-8)
    return np.where(re < 1000.0, 24.0 / re * (1.0 + 0.15 * re ** 0.687), 0.44)


def settling_velocity(diameter_um):
    """重力—阻力平衡下的颗粒沉降速度 w_s (m/s)，供悬移判据使用。

    由 (4/3)(ρs-ρa)g·d/(ρa·Cd) 开方并迭代求 Cd(Re)，
    比 Stokes 公式在 d > 50 μm 时更合理。
    """
    d = np.maximum(np.asarray(diameter_um, dtype=float) * 1e-6, 1e-6)
    w = np.full_like(d, 0.2)
    for _ in range(24):
        re = RHO_AIR * w * d / MU_AIR
        cd = drag_coefficient(re)
        w = np.sqrt((4.0 / 3.0) * (RHO_SAND - RHO_AIR) * G * d / (RHO_AIR * cd))
    return w


def sample_truncated_normal(rng, n, mean_um, std_um, min_um, max_um):
    """拒绝采样生成截断正态粒径，避免负粒径及边界堆积。"""
    mean_um = float(mean_um); std_um = max(float(std_um), 1e-6)
    min_um = float(min_um); max_um = float(max_um)
    if not (0 < min_um < max_um):
        raise ValueError("粒径上下限必须满足 0 < 最小值 < 最大值")
    if not (min_um <= mean_um <= max_um):
        raise ValueError("平均粒径必须位于最小值和最大值之间")
    out = np.empty(int(n), dtype=float)
    filled = 0
    while filled < n:
        batch = rng.normal(mean_um, std_um, max(2 * (n - filled), 64))
        accepted = batch[(batch >= min_um) & (batch <= max_um)]
        take = min(len(accepted), n - filled)
        if take:
            out[filled:filled + take] = accepted[:take]
            filled += take
    return out


# 2.3：纯参数 -> 纯结果的计算函数统一加缓存，重复参数组合瞬时返回
@cache_data(show_spinner="正在模拟沙粒运动…")
def simulate_saltation(u_ref, cover, mean_diameter_um, n_particles, porosity=DEFAULT_POROSITY,
                       veg_height=DEFAULT_VEG_HEIGHT, turbulence_intensity=0.15, seed=7,
                       soil_moisture_pct=0.0, leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                       roughness=DEFAULT_ROUGHNESS, diameter_std_um=40.0,
                       diameter_min_um=80.0, diameter_max_um=320.0,
                       dt=DT, include_lift=True):
    """逐颗粒分状态的沙粒运动：静止（未起动）/ 跃移（弹跳）/ 悬移（随流）。

    2.4 的三项实质改动：
    1) **状态是逐时步演化的，不再只在初始化时判定一次。** 每个粒子每 0.016 s
       按局地等效摩阻速度与局地湍流强度重新判定：跃移颗粒若进入湍流强、风速
       低的植被区，满足 w_s < u*_e(0.8+1.6I_loc) 就转为悬移；悬移颗粒落到近床面
       且沉降占优则转回跃移；静止颗粒在局地 u*_e 超过起动阈值时按速率再起动。
    2) **湍流脉动强度随位置变化**（见 turbulence_scales）：背景项随局地平均风速
       缩放，植被项由冠层剪切层与枝叶尾流给出，因此冠层内 I 显著高于尾流区。
    3) **植被拦截与再起动**：颗粒在冠层内（VEG_START–VEG_END，z≤H）按
       「拦截速率 ∝ 覆盖度×实度×高度」抽样被枝叶捕获并就地沉积；已沉积颗粒
       在局地风速恢复后可按同一阈值再起动。

    轨迹积分增加壁面线性插值：撞击步只积分到 z=0 的精确落点，避免 dt=0.008 s
    时一步跨过床面导致落点与反弹高度失真。
    返回 dict；粒子数不代表真实沙粒总数，轨迹统计不是绝对输沙通量。
    """
    rng = np.random.default_rng(seed)
    n = int(np.clip(n_particles, 10, 1000))
    dt = float(np.clip(dt, 0.0005, 0.02))
    n_steps = max(1, int(round(OBS_WINDOW / dt)))
    snap_every = max(1, int(round(SNAPSHOT_DT / dt)))
    state_every = max(1, int(round(0.016 / dt)))
    diameters_um = sample_truncated_normal(rng, n, mean_diameter_um, diameter_std_um,
                                           diameter_min_um, diameter_max_um)
    d = diameters_um * 1e-6
    area = np.pi * (d / 2.0) ** 2
    masses = RHO_SAND * (4.0 / 3.0) * np.pi * (d / 2.0) ** 3

    z0 = float(np.clip(roughness, 1e-5, 0.05))
    u_star_inlet = friction_velocity(u_ref, z0)
    inlet_u_eff = u_star_inlet * roughness_partition(z0)
    thresholds = moisture_threshold(fluid_threshold(diameters_um), soil_moisture_pct)
    sustain_thresholds = SUSTAIN_RATIO * thresholds
    ti = float(np.clip(turbulence_intensity, 0.01, 0.60))

    settling = settling_velocity(diameters_um)
    suspension_limit = u_star_inlet * (SUSPENSION_BASE + SUSPENSION_TURB * ti)
    started = inlet_u_eff > thresholds
    modes = np.where(~started, MOTION_PINNED,
                     np.where(settling < suspension_limit, MOTION_SUSPENSION,
                              MOTION_SALTATION)).astype(int)

    # 初始状态：床面跃移颗粒群快照，沿 0 ~ 植被前缘分布
    x = rng.uniform(0.5, VEG_START + 5.0, n)
    z = np.full(n, 0.002)
    is_susp = modes == MOTION_SUSPENSION
    n_susp = int(np.count_nonzero(is_susp))
    if n_susp:
        z[is_susp] = rng.uniform(0.08, 0.55, n_susp)
    u_jump = float(wind_profile(0.02, u_ref, z0)[0])
    vx = rng.uniform(0.40, 0.70, n) * u_jump
    vz = rng.uniform(2.0, 5.0, n) * max(inlet_u_eff, 0.05)
    if n_susp:
        vx[is_susp] = rng.uniform(0.85, 1.05, n_susp) * u_jump
        vz[is_susp] = rng.uniform(-0.05, 0.25, n_susp) * max(inlet_u_eff, 0.05)

    active = np.ones(n, dtype=bool)
    ever_moved = started.copy()          # 是否真正离床运动过（决定是否画轨迹）
    deposited = ~started                 # 当前停放在床面
    captured = np.zeros(n, dtype=bool)   # 被植被枝叶拦截
    captured_z = np.full(n, np.nan)      # 拦截发生的高度
    gust = np.zeros(n)                   # 水平 OU 脉动
    vgust = np.zeros(n)                  # 垂向 OU 脉动（悬移颗粒使用）
    first_impact_x = np.full(n, np.nan)

    u_local = np.full(n, u_jump)
    u_eff = np.full(n, inlet_u_eff)
    i_local = np.full(n, ti)
    sigma_u = np.full(n, max(ti * u_jump, 0.05))
    sigma_w = np.full(n, SIGMA_W_RATIO * max(ti * u_jump, 0.05))

    xs_hist = [np.where(active, x, np.nan)]
    zs_hist = [np.where(active, z, np.nan)]

    decay = np.exp(-dt / T_LAGRANGIAN)
    noise_unit = np.sqrt(max(1.0 - decay ** 2, 0.0))
    p_restart = 1.0 - np.exp(-REENTRAIN_RATE * dt)
    p_eff = effective_porosity(porosity, leaf_development)
    solidity = 1.0 - p_eff
    intercept_rate = (INTERCEPT_RATE * float(np.clip(cover, 0.0, 1.0)) * solidity
                      * min(float(veg_height), 6.0) / 2.0)
    p_intercept = 1.0 - np.exp(-intercept_rate * dt)

    for step in range(n_steps):
        ids = np.flatnonzero(active)
        if ids.size == 0:
            break

        # ---- 缓变量：局地风速 / 等效摩阻 / 湍流尺度（每 0.016 s 更新一次）
        if step % state_every == 0:
            z_dyn = np.maximum(z[ids], MIN_DYNAMIC_Z)
            u_local[ids] = local_wind(x[ids], z_dyn, u_ref, cover, porosity, veg_height,
                                      turbulence_intensity, leaf_development, roughness)
            u_eff[ids] = (local_friction_velocity(x[ids], u_ref, cover, porosity, veg_height,
                                                  turbulence_intensity, leaf_development,
                                                  roughness) * roughness_partition(z0))
            su, sw, il = turbulence_scales(x[ids], z_dyn, u_ref, cover, porosity, veg_height,
                                           turbulence_intensity, leaf_development, roughness,
                                           u_local=u_local[ids])
            sigma_u[ids], sigma_w[ids], i_local[ids] = su, sw, il

        moving = ids[modes[ids] != MOTION_PINNED]
        if moving.size:
            is_moving_susp = modes[moving] == MOTION_SUSPENSION
            gust[moving] = (decay * gust[moving]
                            + noise_unit * sigma_u[moving] * rng.normal(size=moving.size))
            susp_m = moving[is_moving_susp]
            if susp_m.size:
                vgust[susp_m] = (decay * vgust[susp_m]
                                 + noise_unit * sigma_w[susp_m] * rng.normal(size=susp_m.size))
            wind = np.maximum(0.0, u_local[moving] + gust[moving])
            rel_x = wind - vx[moving]
            rel_z = np.where(is_moving_susp, vgust[moving] - vz[moving], -vz[moving])
            rel_speed = np.hypot(rel_x, rel_z)
            reynolds = RHO_AIR * rel_speed * d[moving] / MU_AIR
            cd = drag_coefficient(reynolds)
            factor = 0.5 * RHO_AIR * cd * area[moving] * rel_speed
            ax = factor * rel_x / masses[moving]
            az = factor * rel_z / masses[moving] - G

            if include_lift:
                # Saffman 升力：颗粒滞后于气流时受到向上的剪切升力
                z_lift = np.maximum(z[moving], MIN_DYNAMIC_Z)
                bare_z, _ = wind_profile(z_lift, u_ref, z0)
                shear = (u_star_inlet / (KAPPA * z_lift)
                         * np.clip(u_local[moving] / np.maximum(bare_z, 1e-9), 0.0, 1.5))
                lift = (LIFT_COEFF * 6.46 * MU_AIR * (d[moving] / 2.0) ** 2
                        * np.abs(rel_x) * np.sqrt(np.abs(shear) / (MU_AIR / RHO_AIR)))
                az = az + lift / masses[moving]

            vx_new = vx[moving] + ax * dt
            vz_new = vz[moving] + az * dt
            z_new = z[moving] + vz_new * dt
            hit = (z_new <= 0.0) & (vz_new < 0.0)
            # 壁面线性插值：只积分到真实落点，落点与反弹速度不再依赖步长
            frac = np.where(hit,
                            np.clip(z[moving] / np.maximum(-vz_new, 1e-9) / dt, 0.0, 1.0),
                            1.0)
            x[moving] = x[moving] + vx_new * dt * frac
            z[moving] = np.maximum(z[moving] + vz_new * dt * frac, 0.0)
            vx[moving] = vx_new
            vz[moving] = vz_new

            if np.any(hit):
                hit_ids = moving[hit]
                salt_hit = hit_ids[modes[hit_ids] == MOTION_SALTATION]
                susp_hit = hit_ids[modes[hit_ids] == MOTION_SUSPENSION]
                if salt_hit.size:
                    first_mask = np.isnan(first_impact_x[salt_hit])
                    first_impact_x[salt_hit[first_mask]] = x[salt_hit[first_mask]]
                    # 局部维持跃移的能力：植被削减风速 -> 等效 u* 下降 -> 沉积
                    can_sustain = u_eff[salt_hit] > sustain_thresholds[salt_hit]
                    rebound = np.maximum(np.abs(vz[salt_hit]) * 0.35,
                                         2.5 * np.maximum(u_eff[salt_hit], 0.0))
                    z[salt_hit] = 0.002
                    vx[salt_hit] *= 0.90
                    vz[salt_hit] = np.where(can_sustain, rebound, 0.0)
                    stop_ids = salt_hit[~can_sustain]
                    modes[stop_ids] = MOTION_PINNED
                    deposited[stop_ids] = True
                if susp_hit.size:
                    # 悬移颗粒触地后由湍流重新夹带；若沉降已占优则转为跃移
                    z[susp_hit] = 0.005
                    vz[susp_hit] = (np.abs(vz[susp_hit]) * 0.35
                                    + np.abs(vgust[susp_hit]) * 0.6 + 0.05)
                    limit_hit = u_eff[susp_hit] * (SUSPENSION_BASE
                                                   + SUSPENSION_TURB * i_local[susp_hit])
                    modes[susp_hit[settling[susp_hit] >= limit_hit]] = MOTION_SALTATION

        # ---- 状态转换：跃移 ⇄ 悬移，静止 → 跃移（再起动）
        ids = np.flatnonzero(active)
        limit_now = u_eff[ids] * (SUSPENSION_BASE + SUSPENSION_TURB * i_local[ids])
        to_susp = ((modes[ids] == MOTION_SALTATION) & (settling[ids] < limit_now)
                   & (z[ids] > 0.03))
        modes[ids[to_susp]] = MOTION_SUSPENSION
        to_salt = ((modes[ids] == MOTION_SUSPENSION) & (settling[ids] >= limit_now)
                   & (z[ids] < 0.06) & (vz[ids] <= 0.0))
        modes[ids[to_salt]] = MOTION_SALTATION

        resting = ids[(modes[ids] == MOTION_PINNED)
                      & (u_eff[ids] > thresholds[ids]) & (x[ids] < DOMAIN_END)]
        if resting.size:
            pick = resting[rng.random(resting.size) < p_restart]
            if pick.size:
                modes[pick] = MOTION_SALTATION
                z[pick] = 0.002
                vz[pick] = np.maximum(0.6 * inlet_u_eff, 0.08)
                vx[pick] = 0.6 * np.maximum(u_local[pick], 0.05)
                deposited[pick] = False
                ever_moved[pick] = True

        # ---- 植被拦截：冠层内的运动颗粒被枝叶捕获，就地沉积
        if p_intercept > 0:
            in_canopy = (active & (modes != MOTION_PINNED)
                         & (x >= VEG_START) & (x <= VEG_END) & (z <= veg_height))
            cand = np.flatnonzero(in_canopy)
            if cand.size:
                # 叶面积密度近床面最大、向冠层顶部递减：高空悬移颗粒更容易穿过
                z_weight = np.exp(-np.maximum(z[cand], 0.0) / (0.55 * max(float(veg_height), 0.5)))
                pick = cand[rng.random(cand.size) < 1.0 - np.exp(-intercept_rate * dt * z_weight)]
                if pick.size:
                    captured[pick] = True
                    captured_z[pick] = z[pick]
                    deposited[pick] = False      # 拦截不是床面沉积，单独统计
                    modes[pick] = MOTION_PINNED
                    z[pick] = 0.002
                    vz[pick] = 0.0
                    active[pick] = False

        escaped = np.flatnonzero(active & ((x > DOMAIN_END) | (z > 8.0)))
        if escaped.size:
            active[escaped] = False

        if step % snap_every == 0 or step == n_steps - 1:
            snap_x = np.full(n, np.nan)
            snap_z = np.full(n, np.nan)
            snap_x[active] = x[active]
            snap_z[active] = z[active]
            xs_hist.append(snap_x)
            zs_hist.append(snap_z)

    xmat, zmat = np.asarray(xs_hist), np.asarray(zs_hist)
    trajectories = []
    hop_lengths = []
    for i in range(n):
        if not ever_moved[i]:
            continue
        valid = np.isfinite(xmat[:, i]) & np.isfinite(zmat[:, i])
        path_x = xmat[valid, i].tolist()
        path_z = zmat[valid, i].tolist()
        if not path_x or abs(path_x[-1] - x[i]) > 1e-12 or abs(path_z[-1] - z[i]) > 1e-12:
            path_x.append(float(x[i]))
            path_z.append(float(z[i]))
        if path_x:
            trajectories.append((path_x, path_z, float(diameters_um[i]),
                                 float(masses[i]), int(modes[i])))
        if np.isfinite(first_impact_x[i]) and path_x:
            hop_lengths.append(float(max(0.0, first_impact_x[i] - path_x[0])))

    pinned_mask = (modes == MOTION_PINNED) & ~ever_moved & ~captured
    deposit_mask = deposited & ever_moved & ~captured
    return {
        "trajectories": trajectories,
        "hop_lengths": hop_lengths,
        "u_star": float(u_star_inlet),
        "threshold": float(np.median(thresholds)),
        "diameters_um": diameters_um,
        "masses": masses,
        "deposited_x": x[deposit_mask],
        "intercepted_x": x[captured],
        "intercepted_z": captured_z[captured],
        "intercepted": int(np.count_nonzero(captured)),
        "modes": modes,
        "pinned_x": x[pinned_mask],
        "pinned_d": diameters_um[pinned_mask],
        "state_counts": {
            "pinned": int(np.count_nonzero((modes == MOTION_PINNED) & ~captured)),
            "saltation": int(np.count_nonzero(modes == MOTION_SALTATION)),
            "suspension": int(np.count_nonzero(modes == MOTION_SUSPENSION)),
        },
        "suspension_limit": float(suspension_limit),
        "settling_median": float(np.median(settling)),
        "n_eligible": int(np.count_nonzero(started)),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "ti_max": float(np.max(i_local)),
    }


# ------------------------------------------------------------
# 4b. 沿程防风指标（恢复距离、遮蔽积分）
# ------------------------------------------------------------

@cache_data(show_spinner=False)
def recovery_distance(u_ref, cover, porosity, veg_height, z_probe=0.2, recovery_ratio=0.90,
                      turbulence_intensity=0.15,
                      leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                      roughness=DEFAULT_ROUGHNESS):
    """植被带末端下风侧，植被最大风速亏损恢复到 recovery_ratio 的距离。

    例如 0.90 表示剩余亏损不超过植被区最大亏损的 10%。
    返回相对植被带末端的距离；无植被/域内未恢复时返回 None。
    """
    if cover <= 0:
        return None
    x_canopy = np.linspace(VEG_START, VEG_END, 1201)
    bare_canopy, _ = wind_profile(np.full_like(x_canopy, z_probe), u_ref, roughness)
    veg_canopy = local_wind(x_canopy, z_probe, u_ref, cover, porosity, veg_height,
                            turbulence_intensity, leaf_development, roughness)
    max_deficit = float(np.max(bare_canopy - veg_canopy))
    if max_deficit <= max(1e-10, float(np.max(bare_canopy)) * 1e-8):
        return None

    x_down = np.linspace(VEG_END, DOMAIN_END, 15001)
    bare_down, _ = wind_profile(np.full_like(x_down, z_probe), u_ref, roughness)
    veg_down = local_wind(x_down, z_probe, u_ref, cover, porosity, veg_height,
                          turbulence_intensity, leaf_development, roughness)
    remaining_deficit = np.maximum(bare_down - veg_down, 0.0)
    target_deficit = (1.0 - recovery_ratio) * max_deficit
    indices = np.where(remaining_deficit <= target_deficit)[0]
    if len(indices) == 0:
        return None
    return float(x_down[indices[0]] - VEG_END)


def trapezoid_compatible(y, x):
    """兼容 NumPy 1.x / 2.x 的梯形积分。"""
    integrator = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(integrator(y, x))


@cache_data(show_spinner=False)
def porosity_sweep(u_ref, cover, veg_height, turbulence_intensity, z_probe=0.2,
                   leaf_development=DEFAULT_LEAF_DEVELOPMENT,
                   roughness=DEFAULT_ROUGHNESS):
    """扫描孔隙率，以下风侧 60 m 相对风速亏损积分衡量综合遮蔽效果。"""
    porosities = np.linspace(0.0, 0.80, 41)
    x_down = np.linspace(VEG_END, VEG_END + 60.0, 601)
    bare, _ = wind_profile(np.full_like(x_down, z_probe), u_ref, roughness)
    shelter_indices, center_reductions = [], []
    for p_val in porosities:
        veg = local_wind(x_down, z_probe, u_ref, cover, p_val, veg_height,
                         turbulence_intensity, leaf_development, roughness)
        relative_deficit = np.maximum(0.0, (bare - veg) / np.maximum(bare, 1e-9))
        shelter_indices.append(trapezoid_compatible(relative_deficit, x_down))
        center_bare, _ = wind_profile(z_probe, u_ref, roughness)
        center_veg = local_wind((VEG_START + VEG_END) / 2, z_probe, u_ref, cover, p_val,
                                veg_height, turbulence_intensity, leaf_development, roughness)
        center_reductions.append(float((center_bare - center_veg) / max(float(center_bare), 1e-9) * 100.0))
    return porosities, np.asarray(shelter_indices), np.asarray(center_reductions)


# ------------------------------------------------------------
# 5. 输沙能力评价
# ------------------------------------------------------------

def transport_capacity_bagnold(u_star, diameters_um, particle_masses, soil_moisture_pct,
                                roughness, coefficient=1.5):
    """粒径分布加权的简化 Bagnold 型输沙能力估算，单位 kg/(m·s)。

    对每个粒径组使用含水率修正阈值；以代表颗粒质量占比进行加权。
    系数 1.5 为教学默认经验系数，必须通过实测/文献针对具体沙床校准。
    """
    d = np.asarray(diameters_um, dtype=float)
    m = np.asarray(particle_masses, dtype=float)
    if d.size == 0 or m.size == 0 or np.sum(m) <= 0:
        return 0.0
    thresholds = moisture_threshold(fluid_threshold(d), soil_moisture_pct)
    u_eff = max(float(u_star), 0.0) * roughness_partition(roughness)
    excess = np.maximum(u_eff ** 2 - thresholds ** 2, 0.0)
    q_by_particle = coefficient * RHO_AIR / G * u_eff * excess
    mass_weights = m / np.sum(m)
    return float(np.sum(q_by_particle * mass_weights))


@cache_data(show_spinner=False)
def calculate_transport_metrics(u_ref, cover, porosity, veg_height, turbulence_intensity,
                                leaf_development, roughness, soil_moisture_pct,
                                diameters_um, particle_masses, z_probe=0.2,
                                downstream_x=80.0):
    """对比裸地与植被尾流下风侧的粒径加权输沙能力。"""
    _, u_star_bare = wind_profile(z_probe, u_ref, roughness)
    u_down = float(local_wind(downstream_x, z_probe, u_ref, cover, porosity,
                              veg_height, turbulence_intensity, leaf_development, roughness))
    log_term = np.log((z_probe + roughness) / roughness)
    u_star_veg = KAPPA * u_down / max(float(log_term), 1e-9)
    q_bare = transport_capacity_bagnold(u_star_bare, diameters_um, particle_masses,
                                        soil_moisture_pct, roughness)
    q_veg = transport_capacity_bagnold(u_star_veg, diameters_um, particle_masses,
                                       soil_moisture_pct, roughness)
    reduction = ((q_bare - q_veg) / q_bare * 100.0) if q_bare > 0 else None
    return {"q_bare": q_bare, "q_vegetated": q_veg,
            "reduction_pct": None if reduction is None else float(np.clip(reduction, -100, 100)),
            "u_star_bare": float(u_star_bare), "u_star_vegetated": float(u_star_veg),
            "downstream_x": float(downstream_x), "z_probe": float(z_probe)}


