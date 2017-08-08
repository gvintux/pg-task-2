from common.models.kelvin_fought.tension.numeric import tension_state
from common.values import Values

if __name__ == '__main__':
    print('Testing task_1.kelvin_fought.deflection.numeric ...')
    xstp = 1
    ystp = 1
    xrng = range(-200, 200 + 1, xstp)
    yrng = range(0, 1, ystp)

    args = Values()
    args['v'] -= 0.1
    sx, sy, txy = tension_state(args, xrng, yrng)
    file_sx = open("result_py_sx.csv", "w")
    # file_sy = open("result_py_sy.csv", "w")
    # file_txy = open("result_py_txy.csv", "w")
    for key in sorted(sx):
        file_sx.write(str(key) + "," + str(sx[key][0]) + "\n")
        # file_sy.write(str(key) + "," + str(sy[key][0]) + "\n")
        # file_txy.write(str(key) + "," + str(txy[key][0]) + "\n")
    file_sx.close()
    # file_sy.close()
    # file_txy.close()
    print('Test finished')
