import multiprocessing as mp
from cmath import *
from math import exp as rexp

from numpy import inf
from scipy.integrate import nquad


def deflection_func(a):
    k = sqrt(a['l'] ** 2 + a['e'] ** 2)
    C = sin(a['e'] * a['a']) * sin(a['l'] * a['b'])
    phi = a['e'] * a['y'] + a['l'] * (a['x'] - a['v'] * a['t'])
    delta = exp(1j * phi)
    p = 1j * a['freq']
    K = k * tanh(a['H'] * k)

    k0 = 1j * a['D'] * a['t_f'] * a['l'] * a['v'] * (a['e'] ** 4 + k ** 4) + a['D'] * k ** 4 + a['g'] * a['p_w'] - 4 \
                                                        * a['l'] ** 2 * a['v'] ** 2 * (a['h'] * a['p_i'] + a['p_w'] / K)
    k1 = a['D'] * a['t_f'] * k ** 4 + 4j * a['l'] * a['v'] * (a['h'] * a['p_i'] + a['p_w'] / K)
    k2 = a['h'] * a['p_i'] + a['p_w'] / K

    n = 1 / k2
    l = k1 / k2
    m = k0 / k2
    l_half = l / 2
    D = sqrt(l_half ** 2 - m)
    r1 = -l_half - D
    r2 = -l_half + D

    e_r1_t = exp(r1 * a['t'])
    e_r2_t = exp(r2 * a['t'])
    e_p_t = exp(p * a['t'])

    w_i_den = (p - r1) * (p - r2) * (r1 - r2)
    w0 = e_r1_t * (p - r2) + e_r2_t * (r1 - p) + e_p_t * (r2 - r1)
    w0 /= w_i_den

    w1 = r1 * e_r1_t * (p - r2) + r2 * e_r2_t * (r1 - p) + p * e_p_t * (r2 - r1)
    w1 /= w_i_den

    w2 = r1 ** 2 * e_r1_t * (p - r2) + r2 ** 2 * e_r2_t * (r1 - p) + p ** 2 * e_p_t * (r2 - r1)
    w2 /= w_i_den

    denom = a['D'] * k ** 4 + a['g'] * a['p_w'] - 4 * a['l'] ** 2 * a['v'] ** 2 * (
        a['h'] * a['p_i'] + a['p_w'] / K) + 1j * a['D'] * a['t_f'] * a['l'] * a['v'] * (a['e'] ** 4 + k ** 4)

    w1_c = a['D'] * a['t_f'] * k ** 4 + 4j * a['l'] * a['v'] * (a['h'] * a['p_i'] + a['p_w'] / K)
    w2_c = a['h'] * a['p_i'] + a['p_w'] / K
    w = w0 + (1 + w1_c * w1 + w2_c * w2) / denom
    w *= n
    value = -w * delta * C / a['l'] / a['e']
    return value.real


def deflection_state(func, a, xrange, yrange):
    pool = mp.Pool(processes=mp.cpu_count())
    results = dict()
    data = dict()
    for y in yrange:
        results[y] = [pool.apply_async(func=integrate_for, args=(x, y, func, a.copy())) for x in xrange]
    for y in results:
        xlist = results[y]
        for x in xlist:
            x_p, y_p, v = x.get()
            data[x_p, y_p] = v
    return data


def integrator_adapter(func, args, inner_l, inner_u, outer_l, outer_u):
    def R2Func(x, y):
        args['l'] = x
        args['e'] = y
        return func(args)

    return nquad(lambda x, y: R2Func(x, y), [[inner_l, inner_u], [outer_l, outer_u]], full_output=True)[0]


def integrate_for(x, y, func, a):
    a['x'] = x
    a['y'] = y
    a['v'] = 0
    a['freq'] = 0.064516129 * 2
    print(1/a['freq'])
    a['t'] = 45.7/2
    print(str(x) + ';' + str(y))
    return x, y, -4 * a['P'] * integrator_adapter(func, a, 0, inf, 0, inf) / (pi ** 2) / 1000
