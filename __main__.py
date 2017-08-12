import argparse
import sys

from sympy import pprint, init_printing

import common.models.kelvin_fought.deflection.numeric as defnum
import common.models.kelvin_fought.tension.numeric as tennum
from common.models.kelvin_fought.deflection.symbolic import deflection_solve as solve
from common.values import Values

argv = sys.argv
usage_msg = '*usage instructions*'
parser = argparse.ArgumentParser(usage=usage_msg)
parser.add_argument('-s', '--symbolic', action='store_const', const=True, help="symbolic solution")
parser.add_argument('-n', '--numeric', action='store_const', const=True, help="numeric solution")
parser.add_argument('-d', '--deflection', action='store_const', const=True, help="calculate ice deflection")
parser.add_argument('-t', '--tension', action='store_const', const=True, help="calculate ice tension")
parser.add_argument('-sx', '--sigma_x', action='store_const', const=True, help="calculate ice tension sigma_x")
parser.add_argument('-sy', '--sigma_y', action='store_const', const=True, help="calculate ice tension sigma_y")
parser.add_argument('-txy', '--tau_xy', action='store_const', const=True, help="calculate ice tension tau_xy")
parser.add_argument('-x', '--x_range', nargs=3, help="x range")
parser.add_argument('-y', '--y_range', nargs=3, help="y range")
if len(sys.argv) == 1:
    parser.print_help()
args = parser.parse_args(sys.argv[1:])
a = Values()
print(a['freq'])
if args.numeric:
    if args.deflection:
        print('Processing numeric deflection...')
        xb, xe, xs = args.x_range
        xrng = range(int(xb), int(xe) + 1, int(xs))
        yb, ye, ys = args.y_range
        yrng = range(int(yb), int(ye) + 1, int(ys))
        data = defnum.deflection_state(defnum.deflection_func, a, xrng, yrng)
        file = open("deflection_result.csv", "w")
        for point in sorted(data):
            x, y = point
            v = data[point]
            file.write(str(x) + ',' + str(y) + ',' + str(v) + '\n')
        file.close()
        print('Processing numeric deflection finished')

    if args.tension:
        if not (args.sigma_x or args.sigma_y or args.tau_xy):
            print('Processing all numeric tensions...')
            xb, xe, xs = args.x_range
            xrng = range(int(xb), int(xe) + 1, int(xs))
            yb, ye, ys = args.y_range
            yrng = range(int(yb), int(ye) + 1, int(ys))
            sx, sy, txy = tennum.tension_state(a, xrng, yrng, ['sx', 'sy', 'txy'])
            file_sx = open("tension_result_sx.csv", "w")
            file_sy = open("tension_result_sy.csv", "w")
            file_txy = open("tension_result_txy.csv", "w")
            for point in sorted(sx):
                x, y = point
                file_sx.write(str(x) + "," + str(y) + ',' + str(sx[x, y]) + "\n")
                file_sy.write(str(x) + "," + str(y) + ',' + str(sy[x, y]) + "\n")
                file_txy.write(str(x) + "," + str(y) + ',' + str(txy[x, y]) + "\n")
            file_sx.close()
            file_sy.close()
            file_txy.close()
            print('Processing all numeric tensions finished')
        if args.sigma_x:
            print('Processing sigma_x numeric tension...')
            xb, xe, xs = args.x_range
            xrng = range(int(xb), int(xe) + 1, int(xs))
            yb, ye, ys = args.y_range
            yrng = range(int(yb), int(ye) + 1, int(ys))
            sx, sy, txy = tennum.tension_state(a, xrng, yrng, ['sx'])
            file_sx = open("tension_result_sx.csv", "w")
            for point in sorted(sx):
                x, y = point
                file_sx.write(str(x) + "," + str(y) + ',' + str(sx[point]) + "\n")
            file_sx.close()
            print('Processing sigma_x numeric tension finished')
        if args.sigma_y:
            print('Processing sigma_y numeric tension...')
            xb, xe, xs = args.x_range
            xrng = range(int(xb), int(xe) + 1, int(xs))
            yb, ye, ys = args.y_range
            yrng = range(int(yb), int(ye) + 1, int(ys))
            sx, sy, txy = tennum.tension_state(a, xrng, yrng, ['sy'])
            file_sy = open("tension_result_sy.csv", "w")
            for point in sorted(sy):
                x, y = point
                file_sy.write(str(x) + "," + str(y) + ',' + str(sy[point]) + "\n")
            file_sy.close()
            print('Processing sigma_y numeric tension finished')
        if args.tau_xy:
            print('Processing tau_xy numeric tension...')
            xb, xe, xs = args.x_range
            xrng = range(int(xb), int(xe) + 1, int(xs))
            yb, ye, ys = args.y_range
            yrng = range(int(yb), int(ye) + 1, int(ys))
            sx, sy, txy = tennum.tension_state(a, xrng, yrng, ['txy'])
            file_txy = open("tension_result_txy.csv", "w")
            for point in sorted(txy):
                x, y = point
                file_txy.write(str(x) + "," + str(y) + ',' + str(txy[point]) + "\n")
            file_txy.close()
            print('Processing tau_xy numeric tension finished')
if args.symbolic:
    if args.deflection:
        print('Processing symbolic deflection...')
        cols = 80
        init_printing(use_unicode=True, num_columns=cols)
        print("Deflection formula (common solution):")
        pprint(solve())
        print(cols * '=')
        # print("Deflection formula (lambda = 0):")
        # pprint(solve(lm=0))
        # print(cols * '=')
        # print("Deflection formula (eta = 0):")
        # pprint(solve(eta=0))
        # print(cols * '=')
        # print("Deflection formula (lambda = eta = 0):")
        # pprint(solve(lm=0, eta=0))
        # print(cols * '=')
        print('Processing symbolic deflection finished')
    if args.tension:
        print('calc symbolic all tension')
