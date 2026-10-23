"""Лабораторная работа №1, вариант 5 (двойной маятник на тележке): все эксперименты.

Запуск:  python experiments.py
Результат: графики в figures/, численные результаты в results/results.json.
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from lab1 import model as M
from lab1.control import ControlLoop
from lab1.failop import FailOperationalPlanner, ModelEstimator, SensorFaultDetector
from lab1.perception import DelayedSensor, Perception, noisy_sensor
from lab1.planning import design_lqr, linearize, planning_lqr, planning_pd

FIG = M.ROOT / "figures"
RES = M.ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "font.size": 10})

SIM_TIME = 10.0
THETA0 = 0.05          # начальное отклонение первого звена, рад
SIGMA_NOMINAL = 0.0005  # шум исправного энкодера, рад (≈ дискрет 12-битного энкодера)
results = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    print("  ->", name)


def fall_str(tf):
    return "не упал за 10 с" if tf is None else f"{tf:.2f} с"


def architecture_diagram():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.1, 4.2)
    ax.axis("off")
    boxes = [
        (0.2, "Среда (MuJoCo)", "Тележка + 2 звена\nфизика 500 Гц\nmj_step()", "#e8e8e8"),
        (2.9, "Perception", "энкодеры x, θ1, θ2\n+ шум σ, d/dt + ФНЧ\n+ задержка (FIFO)", "#d6e9f8"),
        (5.6, "Planning", "LQR: u = −K(x − x_ref)\nK из уравнения Риккати\n(6 коэффициентов PD)", "#fde9c9"),
        (8.3, "Control", "контур f = 20…1000 Гц\nZOH + насыщение ±40 Н\ndata.ctrl → мотор", "#d9f2d9"),
    ]
    for x, title, body, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.6), 2.4, 1.9, boxstyle="round,pad=0.05", fc=color, ec="k"))
        ax.text(x + 1.2, 3.2, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + 1.2, 2.3, body, ha="center", va="center", fontsize=9)
    for x0, x1 in ((2.6, 2.9), (5.3, 5.6), (8.0, 8.3)):
        ax.add_patch(FancyArrowPatch((x0, 2.55), (x1, 2.55), arrowstyle="-|>", mutation_scale=18))
    ax.add_patch(FancyArrowPatch((9.5, 1.55), (1.4, 1.55), connectionstyle="arc3,rad=-0.15",
                                 arrowstyle="-|>", mutation_scale=18, color="tab:red"))
    ax.text(5.5, 0.12, "сила на тележку F (замкнутый контур, дедлайн такта 2 мс)", ha="center", color="tab:red")
    ax.add_patch(FancyBboxPatch((2.9, 3.65), 5.1, 0.45, boxstyle="round,pad=0.03", fc="#f8d7da", ec="tab:red"))
    ax.text(5.45, 3.87, "Fail-Operational: детектор отказа (дисперсия невязки) → фильтр Калмана по модели",
            ha="center", va="center", fontsize=9)
    ax.set_title("Архитектура «три слоя» для двойного маятника на тележке", fontsize=12)
    save(fig, "fig0_architecture.png")


architecture_diagram()

model, data = M.load()
A, B = linearize(model, data)
K = design_lqr(model, data, 500)
results["model"] = {
    "timestep": model.opt.timestep,
    "physics_hz": 1 / model.opt.timestep,
    "unstable_eigs_continuous": sorted(
        float(np.log(abs(e)) / model.opt.timestep) for e in np.linalg.eigvals(A) if abs(e) > 1.0001
    ),
    "K_lqr_500": K.ravel().round(3).tolist(),
}


# ---------------------------------------------------------------- Часть 1. Без контроллера
print("Эксперимент 1: падение без контроллера")
M.reset(model, data, THETA0)
t, q = [], []
fall_t = None
for step in range(int(5.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)
    t.append((step + 1) * model.opt.timestep)
    q.append(data.qpos.copy())
    if fall_t is None and M.has_fallen(data):
        fall_t = t[-1]
t, q = np.array(t), np.array(q)
a1, a2 = M.absolute_angles(q)
results["baseline"] = {"theta0": THETA0, "fall_time": fall_t}

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, a1, label="Звено 1 (абс. угол)")
ax.plot(t, a2, label="Звено 2 (абс. угол)")
ax.axhline(np.pi / 2, color="k", ls=":", lw=0.8)
ax.axhline(-np.pi / 2, color="k", ls=":", lw=0.8, label="Порог падения ±π/2")
ax.axvline(fall_t, color="r", ls="--", label=f"Падение: {fall_t:.2f} с")
ax.set(xlabel="Время, с", ylabel="Угол от вертикали, рад",
       title=f"Двойной маятник без контроллера (θ1(0) = {THETA0} рад, физика 500 Гц)")
ax.legend()
save(fig, "fig1_baseline.png")

# Нелинейность/хаос: два запуска с разницей 1e-6 рад
print("Эксперимент 1b: чувствительность к начальным условиям")
traj = []
for eps in (0.0, 1e-6):
    M.reset(model, data, THETA0 + eps)
    qq = []
    for _ in range(int(15.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        qq.append(data.qpos.copy())
    traj.append(np.array(qq))
tc = np.arange(1, len(traj[0]) + 1) * model.opt.timestep
diff = np.abs(traj[0][:, 2] - traj[1][:, 2]) + 1e-12
diverge_t = float(tc[np.argmax(diff > 0.1)])
results["chaos"] = {"eps": 1e-6, "divergence_time_0p1rad": diverge_t}

fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axs[0].plot(tc, traj[0][:, 2], label="θ2, θ1(0) = 0.05")
axs[0].plot(tc, traj[1][:, 2], "--", label="θ2, θ1(0) = 0.05 + 1e-6")
axs[0].set(ylabel="Отн. угол θ2, рад", title="Нелинейная динамика: свободное движение двойного маятника")
axs[0].legend(loc="upper left")
axs[1].semilogy(tc, diff)
axs[1].axvline(diverge_t, color="r", ls="--", label=f"|Δθ2| > 0.1 рад при t = {diverge_t:.1f} с")
axs[1].set(xlabel="Время, с", ylabel="|Δθ2|, рад (лог.)")
axs[1].legend()
save(fig, "fig2_chaos.png")


# ---------------------------------------------------------------- Часть 2. Perception
print("Эксперимент 2: истина vs шум vs задержка")
M.reset(model, data, THETA0)
dt = model.opt.timestep
true_l, noisy_l, delayed_l, cam_l = [], [], [], []
delayed = DelayedSensor(delay_steps=10)          # 20 мс
rng = np.random.default_rng(0)
cam_period = round((1 / 30) / dt)               # камера 30 fps
cam_val = 0.0
n = int(0.9 / dt)
for step in range(n):
    mujoco.mj_step(model, data)
    th = data.qpos[1]
    nz = noisy_sensor(th, 0.02, rng)
    true_l.append(th)
    noisy_l.append(nz)
    delayed_l.append(delayed.read(nz))
    if step % cam_period == 0:
        cam_val = nz
    cam_l.append(cam_val)
tp = np.arange(1, n + 1) * dt
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(tp, noisy_l, color="0.75", lw=0.7, label="Шумный энкодер (σ = 0.02 рад, 500 Гц)")
ax.plot(tp, delayed_l, color="tab:orange", lw=1, label="Шум + задержка 20 мс")
ax.step(tp, cam_l, where="post", color="tab:green", lw=1.3, label="«Камера» 30 fps (удержание кадра)")
ax.plot(tp, true_l, color="k", lw=2, label="Истинный угол θ1")
ax.set(xlabel="Время, с", ylabel="Угол θ1, рад", title="Perception: истина vs шум vs задержка vs низкая частота")
ax.legend()
save(fig, "fig3_perception.png")
lag_err = float(np.max(np.abs(np.array(true_l) - np.array(cam_l))))
results["perception"] = {"camera_max_error_rad": lag_err}


# ---------------------------------------------------------------- Часть 3. Planning: PD vs LQR
print("Эксперимент 3: одноконтурный PD vs LQR")
runs = {}
for name, planner in {
    "PD по θ1 (Kp=20, Kd=5)": lambda z, t: planning_pd(z[1], z[4]),
    "PD по θ1 (Kp=200, Kd=20)": lambda z, t: planning_pd(z[1], z[4], Kp=200, Kd=20),
    "LQR (6 переменных состояния)": lambda z, t: planning_lqr(z, K),
}.items():
    M.reset(model, data, THETA0)
    log, tf = ControlLoop(model, 500).run(data, Perception(SIGMA_NOMINAL), planner, sim_time=SIM_TIME)
    runs[name] = (log, tf)
results["pd_vs_lqr"] = {k: v[1] for k, v in runs.items()}
fig, ax = plt.subplots(figsize=(8, 4))
for name, (log, tf) in runs.items():
    ax.plot(log["t"], log["qpos"][:, 1], label=f"{name}: {fall_str(tf)}")
ax.set(xlabel="Время, с", ylabel="Угол θ1, рад", title="Planning: PD по одному углу не держит двойной маятник", xlim=(0, 3))
ax.legend()
save(fig, "fig4_pd_vs_lqr.png")


# ---------------------------------------------------------------- Часть 4. Частота управления
print("Эксперимент 4: частота управления")
# Чтобы 1000 Гц было честным (а не упиралось в шаг физики 2 мс), здесь физика 2 кГц.
mf, df = M.load()
mf.opt.timestep = 0.0005
Kf = design_lqr(mf, df, 500)
freqs = [1000, 500, 200, 100, 50, 20]
freq_rows, freq_logs = [], {}
for f in freqs:
    M.reset(mf, df, THETA0)
    log, tf = ControlLoop(mf, f).run(df, Perception(SIGMA_NOMINAL), lambda z, t: planning_lqr(z, Kf), sim_time=SIM_TIME)
    freq_logs[f] = (log, tf)
    late = log["qpos"][log["t"] > 5.0, 1]
    freq_rows.append({"freq": f, "fall_time": tf, "stable": tf is None,
                      "max_theta_after_5s": float(np.abs(late).max()) if tf is None else None,
                      "u_max": float(np.abs(log["u"]).max())})
# поиск критической частоты
scan_f = list(range(50, 19, -2))
crit = []
for f in scan_f:
    M.reset(mf, df, THETA0)
    _, tf = ControlLoop(mf, f).run(df, Perception(SIGMA_NOMINAL), lambda z, t: planning_lqr(z, Kf), sim_time=SIM_TIME)
    crit.append({"freq": f, "fall_time": tf})
min_stable_f = min(r["freq"] for r in crit if r["fall_time"] is None)
results["frequency"] = {"physics_hz": 2000, "rows": freq_rows, "scan": crit, "min_stable_freq": min_stable_f}

fig, ax = plt.subplots(figsize=(8, 4.5))
for f, (log, tf) in freq_logs.items():
    ax.plot(log["t"], log["qpos"][:, 1], label=f"{f} Гц ({fall_str(tf)})")
ax.set(xlabel="Время, с", ylabel="Угол θ1, рад", title="Влияние частоты управления на устойчивость (LQR, θ1(0)=0.05)",
       xlim=(0, 4), ylim=(-0.6, 0.6))
ax.legend()
save(fig, "fig5_frequency.png")


# ---------------------------------------------------------------- Часть 4.4. Задержка
print("Эксперимент 5: задержка измерений")
delays_ms = [0, 5, 10, 20, 50, 100, 200]
ctrl_dt_ms = 2.0
delay_rows, delay_logs = [], {}
for dms in delays_ms:
    steps = int(np.ceil(dms / ctrl_dt_ms))
    M.reset(model, data, THETA0)
    log, tf = ControlLoop(model, 500).run(data, Perception(SIGMA_NOMINAL, delay_steps=steps),
                                          lambda z, t: planning_lqr(z, K), sim_time=SIM_TIME)
    delay_logs[dms] = (log, tf)
    delay_rows.append({"delay_ms": dms, "delay_steps": steps, "actual_ms": steps * ctrl_dt_ms,
                       "fall_time": tf, "stable": tf is None})
scan_d = []
for dms in range(20, 52, 2):
    M.reset(model, data, THETA0)
    _, tf = ControlLoop(model, 500).run(data, Perception(SIGMA_NOMINAL, delay_steps=dms // 2),
                                        lambda z, t: planning_lqr(z, K), sim_time=SIM_TIME)
    scan_d.append({"delay_ms": dms, "fall_time": tf})
max_ok_delay = max(r["delay_ms"] for r in scan_d if r["fall_time"] is None)
first_fail_delay = min(r["delay_ms"] for r in scan_d if r["fall_time"] is not None)
results["delay"] = {"rows": delay_rows, "scan": scan_d, "max_stable_delay_ms": max_ok_delay,
                    "first_fail_delay_ms": first_fail_delay}

# Задержка + возмущение: тот же маятник, но с толчком 0.15 рад - запас устойчивости меньше
scan_big = []
for dms in range(0, 52, 2):
    M.reset(model, data, 0.15)
    _, tf = ControlLoop(model, 500).run(data, Perception(SIGMA_NOMINAL, delay_steps=dms // 2),
                                        lambda z, t: planning_lqr(z, K), sim_time=SIM_TIME)
    scan_big.append({"delay_ms": dms, "fall_time": tf})
results["delay"]["scan_theta0_0p15"] = scan_big
results["delay"]["max_stable_delay_ms_theta0_0p15"] = max(r["delay_ms"] for r in scan_big if r["fall_time"] is None)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 2]})
for dms, (log, tf) in delay_logs.items():
    axs[0].plot(log["t"], log["qpos"][:, 1], label=f"{dms} мс ({fall_str(tf)})")
axs[0].set(xlabel="Время, с", ylabel="Угол θ1, рад", title="Задержка измерений при 500 Гц", xlim=(0, 3), ylim=(-0.6, 0.6))
axs[0].legend(fontsize=8)
for scan, lab in ((scan_d, "θ1(0) = 0.05 рад"), (scan_big, "θ1(0) = 0.15 рад")):
    xs = [r["delay_ms"] for r in scan]
    ys = [SIM_TIME if r["fall_time"] is None else r["fall_time"] for r in scan]
    axs[1].plot(xs, ys, "o-", ms=4, label=lab)
axs[1].axhline(SIM_TIME, color="k", ls=":", lw=0.8)
axs[1].set(xlabel="Задержка, мс", ylabel="Время до падения, с (10 = устойчив)", title="Граница устойчивости по задержке")
axs[1].legend()
save(fig, "fig6_delay.png")


# ---------------------------------------------------------------- Часть 5. Fail-Operational
print("Эксперимент 6: Fail-Operational")
FAULT_T, FAULT_SIGMA = 2.0, 0.1
K_soft = design_lqr(model, data, 500, Q=np.diag([1, 10, 10, 0.1, 0.1, 0.1]), R=np.array([[10.0]]))
fo_rows = []
fo_logs = {}
for use in (False, True):
    falls, maxang, det_t = [], [], []
    for seed in range(10):
        M.reset(model, data, THETA0)
        det = SensorFaultDetector()
        planner = FailOperationalPlanner(K, det, ModelEstimator(A, B), use_detector=use, K_fault=K_soft)
        per = Perception(SIGMA_NOMINAL, fault_time=FAULT_T, fault_sigma=FAULT_SIGMA, seed=seed)
        log, tf = ControlLoop(model, 500).run(data, per, planner, sim_time=SIM_TIME)
        falls.append(tf)
        det_t.append(det.fault_detected_at)
        if tf is None:
            maxang.append(float(np.abs(log["qpos"][log["t"] > FAULT_T, 1]).max()))
        if seed == 0:
            fo_logs[use] = (log, tf, det.fault_detected_at, np.array(det.var_log))
    fo_rows.append({
        "detector": use,
        "falls": sum(f is not None for f in falls), "runs": len(falls),
        "mean_fall_time": float(np.mean([f for f in falls if f is not None])) if any(f is not None for f in falls) else None,
        "mean_time_after_fault": float(np.mean([f - FAULT_T for f in falls if f is not None])) if any(f is not None for f in falls) else None,
        "max_theta_survived": max(maxang) if maxang else None,
        "detect_delay_ms": float(np.mean([(d - FAULT_T) * 1000 for d in det_t if d is not None])) if use else None,
    })
# ложные срабатывания на исправном сенсоре
false_alarms = 0
for seed in range(10):
    M.reset(model, data, THETA0)
    det = SensorFaultDetector()
    planner = FailOperationalPlanner(K, det, ModelEstimator(A, B), K_fault=K_soft)
    ControlLoop(model, 500).run(data, Perception(SIGMA_NOMINAL, seed=seed), planner, sim_time=SIM_TIME)
    false_alarms += det.faulty
results["failop"] = {"fault_time": FAULT_T, "fault_sigma": FAULT_SIGMA, "rows": fo_rows,
                     "false_alarms": false_alarms, "K_soft": K_soft.ravel().round(3).tolist()}

fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
log0, tf0, _, _ = fo_logs[False]
log1, tf1, det1, var1 = fo_logs[True]
axs[0].plot(log1["t"], log1["meas"][:, 1], color="0.8", lw=0.5, label="Измерение θ1 (σ = 0.1 после отказа)")
axs[0].plot(log0["t"], log0["qpos"][:, 1], color="tab:red", label=f"Без детектора (Fail-Safe): {fall_str(tf0)}")
axs[0].plot(log1["t"], log1["qpos"][:, 1], color="tab:blue", label=f"С детектором (Fail-Operational): {fall_str(tf1)}")
axs[0].axvline(FAULT_T, color="k", ls="--", lw=1, label="Отказ сенсора")
axs[0].set(ylabel="Истинный угол θ1, рад", ylim=(-0.6, 0.6), title="Fail-Operational: отказ энкодера в t = 2 с")
axs[0].legend(fontsize=8, loc="lower left")
axs[1].semilogy(var1[:, 0], var1[:, 1] + 1e-12, label="Дисперсия невязки за 25 тактов")
axs[1].axhline(0.02 ** 2, color="r", ls="--", label="Порог")
axs[1].axvline(det1, color="g", ls=":", label=f"Переключение на модель: {det1:.3f} с")
axs[1].set(xlabel="Время, с", ylabel="Дисперсия, рад²", xlim=(0, 6), ylim=(1e-8, 1))
axs[1].legend(fontsize=8)
save(fig, "fig7_failop.png")


# ---------------------------------------------------------------- Нелинейность: область притяжения
print("Эксперимент 7: область притяжения LQR")
roa = []
for th0 in np.round(np.arange(0.05, 0.61, 0.05), 2):
    M.reset(model, data, float(th0))
    log, tf = ControlLoop(model, 500).run(data, Perception(SIGMA_NOMINAL), lambda z, t: planning_lqr(z, K), sim_time=SIM_TIME)
    roa.append({"theta0": float(th0), "fall_time": tf, "u_sat_share": float(np.mean(np.abs(log["u"]) >= 39.99)),
                "max_cart_x": float(np.abs(log["qpos"][:, 0]).max())})
results["roa"] = roa
fig, ax = plt.subplots(figsize=(7, 3.8))
xs = [r["theta0"] for r in roa]
ax.bar(xs, [r["u_sat_share"] * 100 for r in roa], width=0.03,
       color=["tab:green" if r["fall_time"] is None else "tab:red" for r in roa])
ax.set(xlabel="Начальный угол θ1(0), рад", ylabel="Доля тактов в насыщении ±40 Н, %",
       title="Область притяжения LQR (зелёный - удержал, красный - упал)")
save(fig, "fig8_roa.png")


# ---------------------------------------------------------------- Бюджет времени такта (Edge AI)
print("Эксперимент 8: время вычисления такта")
per = Perception(SIGMA_NOMINAL)
M.reset(model, data, THETA0)
N = 5000
t0 = time.perf_counter()
for i in range(N):
    z = per.read(data, i * 0.002)
    data.ctrl[0] = planning_lqr(z, K)
t_ctrl = (time.perf_counter() - t0) / N
t0 = time.perf_counter()
for i in range(N):
    mujoco.mj_step(model, data)
t_phys = (time.perf_counter() - t0) / N
results["timing"] = {"perception_planning_us": t_ctrl * 1e6, "physics_step_us": t_phys * 1e6, "deadline_us": 2000}

with open(RES / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Готово: results/results.json")
