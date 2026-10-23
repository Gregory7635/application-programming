"""Загрузка модели MuJoCo и вспомогательные функции состояния."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "double_cartpole.xml"

# Индексы в qpos / qvel
X, TH1, TH2 = 0, 1, 2
FALL_ANGLE = np.pi / 2  # критерий падения: звено отклонилось от вертикали больше чем на 90°


def load():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    return model, data


def reset(model, data, theta1=0.05, theta2=0.0, x=0.0):
    """Сброс эпизода. Падение нельзя «откатить» - только начать эпизод заново."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = [x, theta1, theta2]
    mujoco.mj_forward(model, data)


def full_state(data):
    """Истинное состояние [x, th1, th2, x_dot, th1_dot, th2_dot]."""
    return np.concatenate([data.qpos, data.qvel]).copy()


def absolute_angles(q):
    """Абсолютные углы звеньев от вертикали (th2 в модели - относительный)."""
    return q[..., TH1], q[..., TH1] + q[..., TH2]


def has_fallen(data):
    a1, a2 = absolute_angles(data.qpos)
    return abs(a1) > FALL_ANGLE or abs(a2) > FALL_ANGLE
