import multiprocessing as mp
from cmath import *

import numpy as np

from common.models.kelvin_fought.deflection.numeric import deflection_func as w, integrator_adapter


def d2w_dx2_dt(a):
    return -1j * a['l'] ** 3 * a['v'] * dw_dt(a)


def d2w_dy2_dt(a):
    return -1j * a['l'] ** 3 * a['v'] * dw_dt(a)


def dw_dt(a):
    k = sqrt(a['l'] ** 2 + a['e'] ** 2)
    C = sin(a['e'] * a['a']) * sin(a['l'] * a['b'])
    phi = a['e'] * a['y'] + a['l'] * (a['x'] - a['v'] * a['t'])
    delta = exp(-1j * phi)
    p = 1j * a['freq']
    K = k * tanh(a['H'] * k)
    k2 = a['h'] * a['p_i'] + a['p_w'] / K
    k1 = a['D'] * a['t_f'] * k ** 4 + 4j * a['l'] * a['v'] * k2

    n = 1 / k2
    l = k1 / k2
    e_r_t = exp(-l * a['t'])
    e_p_t = exp(p * a['t'])

    w0_a = l * (1 - e_p_t) + p * (1 - e_r_t)
    w0_a *= - 2 * n
    w0_a /= l * p * (l + p)

    w0_b = l ** 2 * (p * a['t'] + 1 + e_p_t) + p ** 2 * (l * a['t'] - 1 + e_r_t)
    w0_b *= n
    w0_b /= l ** 2 * p * (l + p)
    w0 = w0_a + w0_b

    w1_a = (e_p_t - e_r_t)
    w1_a *= 2 * n
    w1_a /= l + p

    w1_b = l * (1 - e_p_t) + p * (1 - e_r_t)
    w1_b *= n
    w1_b /= l * (l + p)
    w1 = w1_a + w1_b

    w2_a = (l * e_r_t + p * e_p_t)
    w2_a *= 2 * n
    w2_a /= l + p

    w2_b = p * (e_p_t - e_r_t)
    w2_b *= -n
    w2_b /= (l + p)

    w2 = w2_a + w2_b

    denom = a['D'] * k ** 4 + a['g'] * a['p_w'] - a['l'] ** 2 * a['v'] ** 2 * (
        k2) + 1j * a['D'] * a['t_f'] * a['l'] * a['v'] * (k ** 4)

    w = (1 + k1 * w1 + k2 * w2) / denom
    w *= a['P_max'] / 2
    value = w * delta * C / a['l'] / a['e']
    return value.real


# def dw1(a):
#     return 1
#
# def dw2(a):
#     return 1


# def w1(a):
#     return 1
#
# def w2(a):
#     return 1


def mx(a):
    phi = a['e'] * a['y'] + a['l'] * (a['x'] - a['v'] * a['t'])
    delta = exp(-1j * phi)
    k = sqrt(a['l'] ** 2 + a['e'] ** 2)
    K = k * tanh(a['H'] * k)
    k2 = a['h'] * a['p_i'] + a['p_w'] / K
    k1 = a['D'] * a['t_f'] * k ** 4 + 4j * a['l'] * a['v'] * k2
    w_val = w(a)
    f1 = -1j * a['l'] ** 3 * a['v'] * w_val
    f2 = -a['e'] ** 4 * w_val

    p = 1j * a['freq']
    n = 1 / k2
    l = k1 / k2
    e_r_t = exp(-l * a['t'])
    e_p_t = exp(p * a['t'])

    # w0_a = l * (1 - e_p_t) + p * (1 - e_r_t)
    # w0_a *= - 2 * n
    # w0_a /= l * p * (l + p)
    #
    # w0_b = l ** 2 * (p * a['t'] + 1 + e_p_t) + p ** 2 * (l * a['t'] - 1 + e_r_t)
    # w0_b *= n
    # w0_b /= l ** 2 * p * (l + p)
    # w0 = w0_a + w0_b

    w1_a = (e_p_t - e_r_t)
    w1_a *= 2 * n
    w1_a /= l + p

    w1_b = l * (1 - e_p_t) + p * (1 - e_r_t)
    w1_b *= n
    w1_b /= l * (l + p)
    w1 = w1_a + w1_b

    w2_a = (l * e_r_t + p * e_p_t)
    w2_a *= 2 * n
    w2_a /= l + p

    w2_b = p * (e_p_t - e_r_t)
    w2_b *= -n
    w2_b /= (l + p)

    w2 = w2_a + w2_b

    dw1_a = 2 * n * (e_p_t - e_r_t)
    dw1_a /= l + p

    dw1_b = n * (l * (1 - e_p_t) + p * (1 - e_r_t))
    dw1_b /= l + p
    dw1 = dw1_a + dw1_b

    dw2_a = 2 * n * (l * e_r_t + p * e_r_t)
    dw2_a /= l + p

    dw2_b = - n * p * (e_p_t - e_r_t)
    dw2_b /= l + p
    dw2 = dw2_a + dw2_b
    diff_num_t = (k1 * dw1 + k2 * dw2 + 1j * a['l'] * a['v'] * (k1 * w1 + k2 * w2 + 1))
    f4 = - a['l'] ** 2 * diff_num_t
    f5 = - a['l'] * a['e'] * diff_num_t
    denom = a['D'] * k ** 4 + a['g'] * a['p_w'] - a['l'] ** 2 * a['v'] ** 2 * (
        k2) + 1j * a['D'] * a['t_f'] * a['l'] * a['v'] * (k ** 4)
    C = sin(a['e'] * a['a']) * sin(a['l'] * a['b'])
    return (C * a['D'] * (f1 + a['mu'] * f2 + a['t_f'] * (f4 + a['mu'] * f5)) / denom * delta / / a['l'] / a['e']).real


def dmy(a):
    return a['D'] * (f2(a) + a['mu'] * f1(a) + a['t_f'] * (f5(a) + a['mu'] * f4(a))) * 16 * a['P'] / (np.pi ** 2) / (
        4 * a['a'] * a['b'])


def dmxy(a):
    return a['D'] * (1 - a['mu']) * (f3(a) + a['t_f'] * f6(a)) * 16 * a['P'] / (np.pi ** 2) / (
        4 * a['a'] * a['b'])


def tension_func_sx(a):
    val = -4 * 6 * a['P_max'] * integrator_adapter(mx, a, 0, np.inf, 0, np.inf) / a['h'] ** 2 / 1000 / (pi ** 2)
    return val


def tension_func_sy(a):
    return 6 * 2 * integrator_adapter(my, a, 0, np.inf, 0, np.inf) / a['h'] ** 2


def tension_func_txy(a):
    return 6 * 2 * integrator_adapter(mxy, a, 0, np.inf, 0, np.inf) / a['h'] ** 2


def tension_state(a, xrange, yrange, specs):
    pool = mp.Pool(processes=mp.cpu_count())
    data_sx = None
    data_sy = None
    data_txy = None
    if 'sx' in specs:
        results = dict()
        data_sx = dict()
        for y in yrange:
            results[y] = [pool.apply_async(func=integrate_for, args=(x, y, tension_func_sx, a.copy())) for x in xrange]
        for y in results:
            xlist = results[y]
            for x in xlist:
                x_p, y_p, v = x.get()
                data_sx[x_p, y_p] = v
    if 'sy' in specs:
        results = dict()
        data_sy = dict()
        for y in yrange:
            results[y] = [pool.apply_async(func=integrate_for, args=(x, y, tension_func_sy, a.copy())) for x in xrange]
        for y in results:
            xlist = results[y]
            for x in xlist:
                x_p, y_p, v = x.get()
                data_sy[x_p, y_p] = v
    if 'txy' in specs:
        results = dict()
        data_txy = dict()
        for y in yrange:
            results[y] = [pool.apply_async(func=integrate_for, args=(x, y, tension_func_txy, a.copy())) for x in xrange]
        for y in results:
            xlist = results[y]
            for x in xlist:
                x_p, y_p, v = x.get()
                data_txy[x_p, y_p] = v
    return data_sx, data_sy, data_txy


def integrate_for(x, y, func, a):
    a['x'] = x
    a['y'] = y
    a['v'] *= 0.99
    print(str(x) + ';' + str(y))
    return x, y, func(a)
