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
    k1 = a['t_f'] * a['D'] * k ** 4 + 3j * a['h'] * a['l'] * a['p_i'] * a['v'] + 4j * a['l'] * a['p_w'] * a['v'] / (
        k * tanh(a['H'] * k))
    l = k1 * n
    k0 = 0
    m = k0 * n
    l_half = l / 2
    # A = a['P_min']
    # B = a['P_max']
    k1 = - l_half + sqrt(l_half ** 2 - m)
    k2 = - l_half - sqrt(l_half ** 2 - m)

    w = 1 / (l * a['freq']) / (l + p) ** 2

    w *= l ** 2 * sin(a['freq'] * a['t']) - 1j * l ** 2 * cos(a['freq'] * a['t']) + l * p * sin(
        a['freq'] * a['t']) + l * a['freq'] * cos(a['freq'] * a['t']) - 2 * l * a['freq'] + l * a['freq'] - p * a[
        'freq'] + p * a['freq']

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
    return x, y, -4 * a['P'] * integrator_adapter(func, a, 0, inf, 0, inf) / (pi ** 2) / 1000
