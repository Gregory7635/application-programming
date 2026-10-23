"""Fail-Operational: детектор отказа сенсора и резервный источник - предсказание по модели."""
from collections import deque

import numpy as np
from scipy.linalg import solve_discrete_are


class ModelEstimator:
    """Резервный источник состояния: предсказание по линейной модели x[k+1] = A x + B u
    с коррекцией по положениям (z = C x, C = [I 0]) через установившийся фильтр Калмана.
    Скорости из сенсора в резервном режиме не используются - их даёт модель.
    Чем больше шум r_meas, тем сильнее оценка опирается на модель."""

    def __init__(self, A, B, q_process=1e-2, r_meas=0.1 ** 2):
        self.A, self.B = A, B
        n = A.shape[0]
        self.nq = n // 2
        self.C = np.hstack([np.eye(self.nq), np.zeros((self.nq, self.nq))])
        # неопределённость модели - в ускорениях (скоростях), положения - интеграл скоростей
        Qn = np.diag([1e-9] * self.nq + [q_process] * self.nq)
        Rn = r_meas * np.eye(self.nq)
        P = solve_discrete_are(A.T, self.C.T, Qn, Rn)
        self.L = P @ self.C.T @ np.linalg.inv(self.C @ P @ self.C.T + Rn)
        self.x = np.zeros(n)

    def init(self, x):
        self.x = np.array(x, dtype=float)

    def predict(self, u):
        self.x = self.A @ self.x + (self.B * u).ravel()
        return self.x

    def correct(self, z):
        self.x = self.x + self.L @ (z[:self.nq] - self.C @ self.x)
        return self.x


class SensorFaultDetector:
    """Отказ = дисперсия невязки углов (измерение минус предсказание модели) за последние
    N тактов больше порога. Берём невязку, а не сырой сигнал, чтобы не путать шум
    с реальным движением маятника.

    Дополнительно - отбраковка единичных выбросов (gate): пока окно копится, подозрительное
    измерение уже не пускается в регулятор."""

    def __init__(self, n_window=25, threshold=0.02 ** 2, gate=0.01, channels=(1, 2)):
        self.window = deque(maxlen=n_window)
        self.threshold = threshold
        self.gate = gate
        self.channels = list(channels)
        self.faulty = False
        self.fault_detected_at = None
        self.var_log = []

    def update(self, residual, t):
        self.window.append(residual[self.channels])
        var = np.var(np.array(self.window), axis=0).max() if len(self.window) > 1 else 0.0
        self.var_log.append((t, var))
        if len(self.window) == self.window.maxlen and var > self.threshold and not self.faulty:
            self.faulty = True
            self.fault_detected_at = t
        return self.faulty

    def is_outlier(self, residual):
        return np.abs(residual[self.channels]).max() > self.gate


class FailOperationalPlanner:
    """Обёртка над LQR: в норме управляет по сенсору, при отказе - по оценке модели."""

    def __init__(self, K, detector, estimator, use_detector=True, u_limit=40.0, K_fault=None):
        self.K = K
        # в деградированном режиме - более «мягкий» регулятор, меньше усиливает ошибку оценки
        self.K_fault = K if K_fault is None else K_fault
        self.detector = detector
        self.est = estimator
        self.use_detector = use_detector
        self.u_limit = u_limit
        self.u_prev = 0.0
        self.started = False
        self.x_hat_log = []

    def __call__(self, z, t):
        if not self.started:
            self.est.init(z)
            self.started = True
        pred = self.est.predict(self.u_prev).copy()
        residual = z - pred
        faulty = self.detector.update(residual, t) if self.use_detector else False
        if faulty:
            x_hat = self.est.correct(z).copy()  # резерв: модель + слабая коррекция по углам
        elif self.use_detector and self.detector.is_outlier(residual):
            x_hat = pred                        # выброс: этот такт живём на предсказании
        else:
            x_hat = np.array(z)                 # норма: доверяем сенсору
            self.est.init(z)
        u = float(-((self.K_fault if faulty else self.K) @ x_hat)[0])
        self.u_prev = float(np.clip(u, -self.u_limit, self.u_limit))
        self.x_hat_log.append(x_hat)
        return u
