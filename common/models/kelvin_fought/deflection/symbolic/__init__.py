from sympy import *
from sympy import pi as PI

D = Symbol('D', real=True, nonzero=True)
pi = Symbol('rho_i', real=True, nonzero=True)
pw = Symbol('rho_w', real=True, nonzero=True)
h = Symbol('h', real=True, nonzero=True)
g = Symbol('g', real=True, nonzero=True)
t = Symbol('t', real=True, positive=True)
q = Symbol('q', real=True, nonzero=True)
r = Symbol('r', real=True, nonzero=True)
mu = Symbol('mu', real=True, nonzero=True)
nu = Symbol('nu', real=True, nonzero=True)
H = Symbol('H', real=True, nonzero=True)
x = Symbol('x', real=True)
y = Symbol('y', real=True)
z = Symbol('z', real=True)
v = Symbol('v', real=True)
tf = Symbol('tau_phi', real=True, nonzero=True)
lm = Symbol('lambda', real=True, nonzero=True)
eta = Symbol('eta', real=True, nonzero=True)
delta = Symbol('delta', real=True)
xi = Symbol('xi', real=True, nonzero=True)
chi = Symbol('chi', real=True, nonzero=True)
a = Symbol('a', real=True, nonzero=True)
b = Symbol('b', real=True, nonzero=True)
A = Symbol('A', real=True, nonzero=True)
B = Symbol('B', real=True, nonzero=True)
dPhidt = Symbol('dPhi_dt', real=True, nonzero=True)
P = Symbol('P', real=True, nonzero=True)
k = Symbol('k')
freq = Symbol('omega', real=True, positive=True)
tau = Symbol('tau', real=True)
w1 = Function('w')(x, y)
w2 = Function('w')(t)
w = w1 + w2
w = Function('w')(x, y, t)
Phi1 = Function('Phi')(x, y, z)
Phi2 = Function('Phi')(t)
Phi = Phi1 + Phi2
Phi = Function('Phi')(x, y, z, t)


def nabla4(func):
    d4_dx4 = diff(func, x, 4)
    d4_dy4 = diff(func, y, 4)
    d4_dx2_dy2 = diff(diff(func, x, 2), y, 2)
    return d4_dx4 + 2 * d4_dx2_dy2 + d4_dy4


def fourier_integral(func, dlt):
    return 1 / (2 * PI) * Integral(func * dlt, (lm, -oo, +oo), (eta, -oo, +oo))


def deflection_solve(**specs):
    if 'lm' in specs:
        global lm
        lm = 0
    if 'eta' in specs:
        global eta
        eta = 0
    delta = exp(-I * (lm * (x - v * t) + eta * y))
    model = Eq(D * (nabla4(w) + tf * diff(nabla4(w), t)) + pw * g * w + pi * h * diff(w, t, 2) + pw * diff(Phi, t), P)
    print("General model")
    pprint(model)
    u = Symbol('u')
    ss = u - v * t
    model = model.subs(x, ss).doit()
    model = model.replace(ss, x).doit()
    # model = model.subs(w, w1 + w2).subs(Phi, Phi1 + Phi2).doit()
    # model = model.subs(w2, 0).subs(Phi2, 0).doit()
    Phi_xyz = Function('Phi')(x, y, z)
    Phi_t = Function('Phi')(t)
    w_xy = Function('w')(x, y)
    w_t = Function('w')(t)
    # model = model.replace(w1, w_xy).replace(Phi1, Phi_xyz).doit()
    # model = model.replace(w2, w_t).replace(Phi2, Phi_t).doit()
    print("Model with x := x - v*t substitution")
    pprint(model)
    w_le = Function('w')(lm, eta)
    Phi_le = Function('Phi')(lm, eta)
    w_let = Function('w')(lm, eta, t)
    Phi_let = Function('Phi')(lm, eta, t)

    # Fourier expressions
    w_f = w_let * delta
    k = Symbol('k')
    Phi_f = (Phi_let * cosh((H + z) * k)) * delta
    laplace_rule = Eq(diff(Phi_f, x, 2) + diff(Phi_f, y, 2) + diff(Phi_f, z, 2), 0)
    pprint("Laplace rule")
    pprint(laplace_rule)
    # ?analyze another solutions
    k_slv = solve(laplace_rule, k)
    print("k solutions")
    pprint(k_slv)
    if lm == 0 and eta != 0:
        k_slv = k_slv[1]
    if eta == 0 and lm != 0:
        k_slv = k_slv[0]
    if eta == 0 and lm == 0:
        k_slv = k_slv[0]
    if eta != 0 and lm != 0:
        k_slv = k_slv[2]

    Phi_f = Phi_f.subs(k, k_slv).doit()
    # ice-water line z = 0
    iw_line = Eq(diff(w, t).doit(), diff(Phi, z))
    print('Ice-water border equation')
    pprint(iw_line)
    iw_line = iw_line.subs(x, ss).doit()
    iw_line = iw_line.replace(ss, x).doit()
    print('Ice-water border equation with x := x - v*t substitution')
    pprint(iw_line)
    # iw_line = iw_line.subs(w, w1 + w2).subs(Phi, Phi1 + Phi2).doit()
    # print('w=w1+w2')
    # pprint(iw_line)
    # iw_line = iw_line.subs(w2, w_t * delta).subs(Phi2, Phi_t * delta).doit()
    # print('w2 = w(t)')
    # pprint(iw_line)
    # iw_line = iw_line.replace(w1, w_le).replace(Phi1, Phi_le).doit()
    # print('w1 = w(l,e)')
    # pprint(iw_line)
    # iw_line_f = iw_line.subs(Phi1, Phi_f).doit().subs(w1, w_f).doit()
    iw_line_f = iw_line.subs(Phi, Phi_f).doit().subs(w, w_f).subs(z, 0).doit()
    print("Ice-water border after subs")
    pprint(iw_line_f)
    Phi_let_slv = None
    try:
        Phi_let_slv = solve(iw_line_f, Phi_let)[0]
    except TypeError:
        Phi_let_slv = Number('0')
    except IndexError:
        Phi_let_slv = Number('0')
    pprint('Phi(lambda, eta, t) solution')
    pprint(Phi_let_slv)
    Phi_f_slv = Phi_f.subs(Phi_let, Phi_let_slv).doit().subs(z, 0).simplify()
    pprint('Phi(x, y, z, t) solution')
    pprint(Phi_f_slv)
    model_f = model.subs(Phi, Phi_f_slv).subs(w, w_f).subs(P, P * delta).doit()
    w_let_slv = solve(model_f, w_let)[0].doit()
    w_f_slv = w_f.subs(w_let, w_let_slv).doit()
    K = tanh(H * k_slv) * k_slv
    K_sym = Symbol('K')
    pprint('w(x, y, t) solution')
    w_let_slv = w_let_slv.subs(K, K_sym).doit().simplify()
    pprint(w_let_slv)
    w_lap = Symbol('W')
    w_let_slv = w_let_slv.subs(diff(w_let, t, 2), tau ** 2 * w_lap).subs(diff(w_let, t), tau * w_lap).subs(w_let,
                                                                                                           w_lap).doit()
    w_lap_slv = solve(w_let_slv, w_lap)[0].expand().collect(tau ** 2).collect(tau)
    print('w_lap solution')
    pprint(w_lap_slv)
    num, den = fraction(w_lap_slv)
    num /= K_sym
    den /= K_sym
    P_lap = laplace_transform(P * exp(I * freq * t) * Heaviside(t), t, tau)[0]
    num = num.subs(P, P_lap).doit()
    print('P laplace')
    pprint(num)
    den = den.expand().collect(tau ** 2).collect(tau)
    print('denom')
    pprint(den)
    k1, k2, k3 = symbols('k1 k2 k3')
    den = den.subs(den.coeff(tau ** 2), k3).subs(den.coeff(tau), k2).doit()
    den /= k3
    l, m = symbols('l m', positive=True)
    print('denom after coeff subs')
    den = den.expand().subs(k2 / k3, l).subs(k1 / k3, m).doit().simplify()
    pprint(den)
    w_lap_slv = inverse_laplace_transform(num / den, tau, t)
    pprint(w_lap_slv.simplify(ratio=oo))
    return w_lap_slv.simplify(ratio=oo)
