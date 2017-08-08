from common.models.kelvin_fought.tension.numeric import *
from common.values import Values

if __name__ == '__main__':
    print('Testing task_1.kelvin_fought.tension.numeric ...')
    args = Values()
    args['l'] = 0.1
    args['e'] = 0.1
    args['t'] = 1
    args['x'] = 10
    args['v'] -= 0.1
    vals = [dmx(args), dmy(args), dmxy(args)]
    print(vals)
    vals = tension_func_sx(args)
    print(vals)
    print('Test finished')
