"""Слой Perception: извлечение состояния, шум сенсоров, задержка измерения.

Модель сенсоров как на реальном роботе: энкодеры измеряют только положения
(x тележки, углы th1, th2), а скорости получаются численным дифференцированием
положений с фильтром низких частот. Поэтому шум в углах усиливается в скоростях
примерно в sqrt(2)/dt раз - главная причина, почему шумный энкодер опасен.
"""
from collections import deque

import numpy as np

from .model import full_state


def noisy_sensor(true_value, sigma=0.02, rng=np.random):
    return true_value + rng.normal(0, sigma, size=np.shape(true_value))


class DelayedSensor:
    """FIFO-буфер: возвращает измерение, сделанное delay_steps тактов назад.

    В методичке maxlen=delay_steps даёт задержку delay_steps-1 такта,
    поэтому здесь буфер на один элемент длиннее.
    """

    def __init__(self, delay_steps):
        self.buffer = deque(maxlen=delay_steps + 1)

    def read(self, value):
        self.buffer.append(value)
        return self.buffer[0]


class Perception:
    """Сенсорный блок: истина -> шум энкодеров -> оценка скорости -> задержка.

    sigma - СКО шума энкодеров (м для x, рад для углов).
    После fault_time (отказ сенсора) СКО становится fault_sigma.
    velocity="diff" - скорости дифференцированием (по умолчанию), "true" - идеальные из qvel.
    """

    def __init__(self, sigma=0.0, delay_steps=0, fault_time=None, fault_sigma=0.1,
                 velocity="diff", cutoff_hz=50.0, seed=0):
        self.sigma = sigma
        self.fault_time = fault_time
        self.fault_sigma = fault_sigma
        self.velocity = velocity
        self.tau = 1.0 / (2 * np.pi * cutoff_hz)
        self.delay = DelayedSensor(delay_steps)
        self.rng = np.random.default_rng(seed)
        self.prev_pos = None
        self.prev_t = None
        self.vel = None

    def current_sigma(self, t):
        if self.fault_time is not None and t >= self.fault_time:
            return self.fault_sigma
        return self.sigma

    def read(self, data, t):
        true = full_state(data)
        nq = len(true) // 2
        sigma = self.current_sigma(t)
        pos = noisy_sensor(true[:nq], sigma, self.rng) if sigma > 0 else true[:nq].copy()

        if self.velocity == "true":
            vel = true[nq:]
        elif self.prev_pos is None:
            vel = np.zeros(nq)
        else:
            dt = t - self.prev_t
            raw = (pos - self.prev_pos) / dt
            alpha = dt / (self.tau + dt)          # ФНЧ первого порядка
            vel = self.vel + alpha * (raw - self.vel)
        self.prev_pos, self.prev_t, self.vel = pos, t, vel
        return self.delay.read(np.concatenate([pos, vel]))
