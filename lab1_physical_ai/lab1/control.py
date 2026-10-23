"""Слой Control: контур управления с фиксированной частотой поверх физики 500 Гц."""
import mujoco
import numpy as np

from .model import has_fallen


class ControlLoop:
    """Perception -> Planning -> Control с частотой freq_hz.

    Физика шагает с model.opt.timestep (2 мс); команда пересчитывается раз в
    steps_per_control шагов и между пересчётами удерживается (ZOH), как в реальном
    приводе. Насыщение - ctrlrange актуатора (±40 Н).
    """

    def __init__(self, model, freq_hz=500):
        self.model = model
        self.sim_dt = model.opt.timestep
        self.steps_per_control = max(1, round(1.0 / freq_hz / self.sim_dt))
        self.dt = self.steps_per_control * self.sim_dt
        self.u_min, self.u_max = model.actuator_ctrlrange[0]

    def run(self, data, perception, planner, sim_time=10.0):
        """planner(measurement, t) -> u. Возвращает лог и время падения (None, если не упал)."""
        n_steps = int(round(sim_time / self.sim_dt))
        log = {"t": [], "qpos": [], "qvel": [], "u": [], "meas": []}
        meas = None
        for step in range(n_steps):
            t = step * self.sim_dt
            if step % self.steps_per_control == 0:
                meas = perception.read(data, t)
                u = planner(meas, t)
                data.ctrl[0] = np.clip(u, self.u_min, self.u_max)
            mujoco.mj_step(self.model, data)
            log["t"].append(t + self.sim_dt)
            log["qpos"].append(data.qpos.copy())
            log["qvel"].append(data.qvel.copy())
            log["u"].append(float(data.ctrl[0]))
            log["meas"].append(np.array(meas))
            if has_fallen(data):
                return _as_arrays(log), t + self.sim_dt
        return _as_arrays(log), None


def _as_arrays(log):
    return {k: np.array(v) for k, v in log.items()}
