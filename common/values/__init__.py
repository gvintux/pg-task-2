from math import sqrt


class Values:
    __a = {
        't_f': 10,
        'p_i': 900.0,
        'g': 9.81,
        'h': 0.5,
        'E': 5e9,
        'mu': 0.3,
        'H': 5.0,
        'p_w': 1000.0,
        'a': 15,
        'b': 6.5,
        'P': 400000,
        'x': 0,
        'y': 0,
        'l': 0,
        'e': 0,
        't': 0,
        'v': None,
        'D': None,
        'freq': None,
        'P_min': 0,
        'P_max': 400000,
        'phi': 0
    }

    def __init__(self):
        a = self.__a
        a['v'] = sqrt(a['g'] * a['H'])
        a['D'] = a['E'] * a['h'] ** 3 / (12 * (1 - a['mu'] ** 2))
        a['freq'] = a['g'] * a['H'] * sqrt(a['p_i'] * a['h'] / a['D'])

    def __getitem__(self, item):
        return self.__a[item]

    def __setitem__(self, key, value):
        a = self.__a
        if key == 'D':
            pass
        elif key == 'E' or key == 'h' or key == 'mu':
            a[key] = value
            a['D'] = a['E'] * a['h'] ** 3 / (12 * (1 - a['mu'] ** 2))
            a['freq'] = a['g'] * a['H'] * sqrt(a['p_i'] * a['h'] / a['D'])
        else:
            a[key] = value

    def copy(self):
        c = Values()
        c.__a = self.__a.copy()
        return c

    def __str__(self):
        return self.__a.__str__()
