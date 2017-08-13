import multiprocessing as mp
from cmath import *

from numpy import inf
from scipy.integrate import nquad


def deflection_func(a):
    k = sqrt(a['l'] ** 2 + a['e'] ** 2)
    C = sin(a['e'] * a['a']) * sin(a['l'] * a['b'])
    phi = a['e'] * a['y'] + a['l'] * (a['x'] - a['v'] * a['t'])
    delta = exp(1j * phi)
    p = 1j * a['freq']
    k2 = a['h'] * a['p_i'] + a['p_w'] / (k * tanh(a['H'] * k))
    n = 1 / k2
    k1 = a['D'] * a['e'] ** 4 * a['t_f'] + 2 * a['D'] * a['e'] ** 2 * a['l'] ** 2 * a['t_f'] + a['D'] * a['l'] ** 4 * a[
        't_f'] + 3j * a['h'] * a['l'] * a['p_i'] * a['v'] + 4j * a['l'] * a['p_w'] * a['v'] / (k * tanh(a['H'] * k))
    l = k1 * n
    k0 = 2j * a['D'] * a['e'] ** 4 * a['l'] * a['t_f'] * a['v'] + a['D'] * a['e'] ** 4 + 2j * a['D'] * a['e'] ** 2 * a[
                                                                                                                         'l'] ** 3 * \
                                                                                         a['t_f'] * a['v'] + 2 * a[
        'D'] * a['e'] ** 2 * a['l'] ** 2 + 1j * a['D'] * a['l'] ** 5 * a['t_f'] * a['v'] + a['D'] * a['l'] ** 4 + a[
             'g'] + a['p_w'] - 3 * a['h'] * a['l'] ** 2 * a['p_i'] * a['v'] ** 2 - 3 * a['l'] ** 2 * a['p_w'] * a[
                                                                                                                    'v'] ** 2 / (
                                                                                       k * tanh(a['H'] * k))
    m = k0 * n
    l_half = l / 2
    A = a['P_min']
    B = a['P_max']
    k1 = - l_half + sqrt(l_half ** 2 - m)
    k2 = - l_half - sqrt(l_half ** 2 - m)
    w = (A - B) * exp(1j*(a['freq'] * a['t'] + a['phi'])) / (k1 - p) / (p - k2)
    w -= exp(k1*a['t']) * (A*k1*exp(1j*a['phi']) - A*k1 + A*p  -B*k1*exp(1j*a['phi']) - B*k1 + B*p) / (k1) / (k1-k2) / (k1 - p)
    w -= exp(k2*a['t']) * (A*k2*exp(1j*a['phi']) - A*k2 + A*p  -B*k2*exp(1j*a['phi']) - B*k + B*p) / (k2) / (k2-k1) / (k2 - p)
    w += (A + B) / (k1*k2)
    w *= n/2
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
    a['t'] = 4
    print(str(x) + ';' + str(y))
    return x, y, -16 * integrator_adapter(func, a, 0, inf, 0, inf) / (pi * 2) / 10000
