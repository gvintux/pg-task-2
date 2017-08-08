from sys import float_info

from common.models.kelvin_fought.deflection.numeric import deflection_func
from common.values import Values

_eps = float_info[8]

if __name__ == '__main__':
    print('Testing task_1.kelvin_fought.deflection.numeric ...')
    args = Values()
    args['l'] = 0.1
    args['e'] = 0.1
    args['t'] = 1
    args['x'] = 10
    val = deflection_func(args)
    print(val)
    # target = 0
    # assert abs(val - target) < _eps, 'Wrong deflection function behavior'
    # target = 9.129897312756954e-05
    # args['l'] = 0.1
    # args['e'] = 0
    # val = value(args)
    # assert abs(val - target) < _eps, 'Wrong deflection function behavior'
    # target = 0.004153803060285196
    # args['l'] = 0
    # args['e'] = 0.1
    # val = value(args)
    # assert abs(val - target) < _eps, 'Wrong deflection function behavior'
    # target = -5.032193553819958e-05
    # args['l'] = 0.1
    # args['e'] = 0.1
    # val = value(args)
    # assert abs(val - target) < _eps, 'Wrong deflection function behavior'
    print('Test finished')
