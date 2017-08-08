import multiprocessing as mp
from math import *

from numpy import inf
from scipy.integrate import nquad


def deflection_func(a):
    k = sqrt(a['l'] ** 2 + a['e'] ** 2)
    C = sin(a['e'] * a['a']) * sin(a['l'] * a['b'])
    phi = a['e'] * a['y'] + a['l'] * (a['x'] - a['v'] * a['t'])
    if a['l'] == 0 and a['e'] != 0:
        A = a['D'] * a['e'] ** 4 + a['g'] * a['p_w']
        B = 0
        return a['b'] * C * (A * cos(phi) - B * sin(phi)) / a['e'] / (A ** 2 + B ** 2)
    if a['l'] != 0 and a['e'] == 0:
        A = a['g'] * a['p_w']
        B = 0
        return a['a'] * a['b'] * A / (A ** 2 + B ** 2)
    if a['l'] == 0 and a['e'] == 0:
        A = a['g'] * a['p_w']
        B = 0
        return a['a'] * a['b'] * A / (A ** 2 + B ** 2)
    A = a['D'] * k ** 4 + a['g'] * a['p_w'] - a['l'] ** 2 * a['v'] ** 2 * (
        a['h'] * a['p_i'] + a['p_w'] / (k * tanh(a['H'] * k)))
    B = a['D'] * a['l'] * a['t_f'] * a['v'] * a['e'] ** 4
    return (A * cos(phi) - B * sin(phi)) * C / a['l'] / a['e'] / (A ** 2 + B ** 2)


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
    print(str(x) + ';' + str(y))
    return x, y, -16 * a['P'] * integrator_adapter(func, a, 0, inf, 0, inf) / (pi * 2) / 10000
