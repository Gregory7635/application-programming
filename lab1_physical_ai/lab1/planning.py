"""Слой Planning: линеаризация модели и LQR (многоконтурный PD) для двойного маятника.

Одноконтурный PD по углу первого звена (как в базовом варианте cart-pole) двойной
маятник не удерживает: у системы два неустойчивых полюса, и обратная связь нужна
по всем шести переменным состояния. LQR даёт именно это - матрицу коэффициентов
K = [Kp_x, Kp_th1, Kp_th2, Kd_x, Kd_th1, Kd_th2], т.е. PD-регулятор, коэффициенты
которого получены решением уравнения Риккати, а не подбором вручную.
"""
import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from .model import reset

# Веса LQR: [x, th1, th2, x_dot, th1_dot, th2_dot]
Q_DEFAULT = np.diag([10.0, 100.0, 100.0, 1.0, 1.0, 1.0])
R_DEFAULT = np.array([[0.1]])


def linearize(model, data):
    """Дискретная линеаризация x[k+1] = A x[k] + B u[k] в верхнем положении (шаг = timestep)."""
    reset(model, data, theta1=0.0, theta2=0.0)
    data.ctrl[:] = 0.0
    nx = 2 * model.nv
    A = np.zeros((nx, nx))
    B = np.zeros((nx, model.nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    return A, B


def resample(A, B, n):
    """Модель для периода управления n * timestep с удержанием управления (ZOH)."""
    An = np.linalg.matrix_power(A, n)
    Bn = sum(np.linalg.matrix_power(A, i) for i in range(n)) @ B
    return An, Bn


def dlqr(A, B, Q=Q_DEFAULT, R=R_DEFAULT):
    P = solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def design_lqr(model, data, control_freq_hz=500, Q=Q_DEFAULT, R=R_DEFAULT):
    A, B = linearize(model, data)
    n = max(1, round(1.0 / control_freq_hz / model.opt.timestep))
    Ad, Bd = resample(A, B, n)
    return dlqr(Ad, Bd, Q, R)


def planning_lqr(state, K, target=None):
    """u = -K (x - x_ref). target - желаемое состояние (по умолчанию вертикаль в x = 0)."""
    if target is None:
        target = np.zeros_like(state)
    return float(-(K @ (state - target))[0])


def planning_pd(theta, theta_dot, target=0.0, Kp=20.0, Kd=5.0):
    """Базовый PD из методички (вариант 1). Для двойного маятника недостаточен.

    Знак: при наклоне маятника вперёд (theta > 0) тележку надо толкать вперёд.
    """
    return Kp * (theta - target) + Kd * theta_dot
