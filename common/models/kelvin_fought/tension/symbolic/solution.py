from sympy import pprint, init_printing

from common.models.kelvin_fought.deflection.symbolic import deflection_solve as solve

if __name__ == '__main__':
    cols = 80
    init_printing(use_unicode=True, num_columns=cols)
    print("Deflection formula (general):")
    pprint(solve())
    print(cols * '=')
    pprint("Deflection formula (lambda = 0):")
    pprint(solve(lm=0))
    print(cols * '=')
    pprint("Deflection formula (eta = 0):")
    pprint(solve(eta=0))
    print(cols * '=')
    pprint("Deflection formula (lambda = eta = 0):")
    pprint(solve(lm=0, eta=0))
    print(cols * '=')
