import multiprocessing as mp
from cmath import *

from numpy import inf
from scipy.integrate import nquad


def deflection_func(a):
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
    a['v'] *= 0.99
    # a['v'] = 0
    T = 1 / a['freq'] * 2 * pi
    # a['t'] = T - 1 / a['freq']
    a['t'] = 1 / a['freq'] / 2
    print(str(x) + ';' + str(y))
    return x, y, -4 * integrator_adapter(func, a, 0, inf, 0, inf) / (pi ** 2) / 1000
