from sympy import *
from sympy import pi as PI

D = Symbol('D', real=True, nonzero=True)
pi = Symbol('rho_i', real=True, nonzero=True)
pw = Symbol('rho_w', real=True, nonzero=True)
h = Symbol('h', real=True, nonzero=True)
g = Symbol('g', real=True, nonzero=True)
t = Symbol('t', real=True, nonzero=True)
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

w1 = Function('w1')(x, y)
w2 = Function('w2')(t)
w = w1 + w2
w = Function('w')(x, y, t)
Phi1 = Function('Phi1')(x, y, z)
Phi2 = Function('Phi2')(t)
Phi = Phi1 + Phi2
Phi = Function('Phi')(x, y, z, t)


def nabla4(func):
    d4_dx4 = diff(func, x, 4)
    d4_dy4 = diff(func, y, 4)
    d4_dx2_dy2 = diff(diff(func, x, 2), y, 2)
    return d4_dx4 + 2 * d4_dx2_dy2 + d4_dy4


def diff_t(func):
    return diff(func, t) - v * diff(func, x)


def diff_t2(func):
    return diff(func, t, 2) - 2 * v * diff(diff(func, x), t) + v ** 2 * diff(func, x, 2)


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
    model = D * (nabla4(w) + tf * diff(nabla4(w), t)) + pw * g * w + pi * h * diff(w, t, 2) + pw * diff(Phi, t) - P
    print("General model")
    pprint(model)
    u = Symbol('u')
    ss = u - v * t
    model = model.subs(x, ss).doit()
    model = model.replace(ss, x).doit()
    model = model.subs(w, w1 + w2).subs(Phi, Phi1 + Phi2).doit()
    model = model.subs(w2, 0).subs(Phi2, 0).doit()
    Phi_xyz = Function('Phi')(x, y, z)
    w_xy = Function('w')(x, y)
    model = model.replace(w1, w_xy).replace(Phi1, Phi_xyz).doit()
    print("Model with x := x - v*t substitution")
    pprint(model)
    w_le = Function('w')(lm, eta)
    Phi_le = Function('Phi')(lm, eta)
    # Fourier expressions
    w_f = w_le * delta
    Phi_f = Phi_le * cosh((H + z) * k) * delta
    laplace_rule = Eq(diff(Phi_f, x, 2) + diff(Phi_f, y, 2) + diff(Phi_f, z, 2), 0)
    # ?analyze another solutions
    k_slv = solve(laplace_rule, k)
    print("k solutions")
    pprint(k_slv)
    if lm == 0 and eta != 0:
        Phi_f = Phi_f.subs(k, k_slv[1]).doit()
    if eta == 0 and lm != 0:
        Phi_f = Phi_f.subs(k, k_slv[0]).doit()
    if eta == 0 and lm == 0:
        Phi_f = Phi_f.subs(k, k_slv[0]).doit()
    if eta !=0 and lm !=0:
        Phi_f = Phi_f.subs(k, k_slv[2]).doit()
    # ice-water line z = 0
    iw_line = Eq(diff(w, t).doit(), diff(Phi, z))
    print('Ice-water border equation')
    pprint(iw_line)
    iw_line = iw_line.subs(x, ss).doit()
    iw_line = iw_line.replace(ss, x).doit()
    print('Ice-water border equation with x := x - v*t substitution')
    pprint(iw_line)
    iw_line = iw_line.subs(w, w1 + w2).subs(Phi, Phi1 + Phi2).doit()
    iw_line = iw_line.subs(w2, 0).subs(Phi2, 0).doit()
    iw_line = iw_line.replace(w1, w_xy).replace(Phi1, Phi_xyz).doit()
    pprint(iw_line)
    iw_line_f = iw_line.subs(Phi_xyz, Phi_f).doit().subs(w_xy, w_f).doit()
    pprint(iw_line_f)
    Phi_le_slv = None
    try:
        Phi_le_slv = solve(iw_line_f, Phi_le)[0]
    except TypeError:
        Phi_le_slv = Number('0')
    except IndexError:
        Phi_le_slv = Number('0')
    pprint('Phi(lambda, eta) solution')
    pprint(Phi_le_slv)
    Phi_f_slv = Phi_f.subs(Phi_le, Phi_le_slv).doit().subs(z, 0).simplify()
    pprint('Phi(x, y, z, t) solution')
    pprint(Phi_f_slv)
    w_le_slv = model.subs(Phi_xyz, Phi_f_slv).subs(w_xy, w_f).subs(P, P * delta).doit()
    w_le_slv = solve(w_le_slv, w_le)[0]
    pprint('w(x, y, t) solution')
    pprint(w_le_slv)
    numer, denom = fraction(w_le_slv)
    K = numer.coeff(P)
    numer /= K
    denom = (denom / K).simplify()
    denom_im = im(denom).collect(D)
    R = factor(denom_im.coeff(D).simplify(), deep=True)
    denom_im = (denom_im - R * D).simplify() + R * D
    denom_re = re(denom).collect(D)
    T = factor(denom_re.coeff(D).simplify(), deep=True)
    denom_re = (denom_re - T * D).simplify() + T * D
    Aa = denom_re
    Bb = denom_im
    conj = A - I * B
    denom = ((A + I * B) * conj).simplify()
    w_le_slv_simp = (numer / denom)
    w_slv = w_f.subs(w_le, w_le_slv_simp).doit()
    pprint('w(x, y, t) solution')
    pprint(w_slv)
    # load size
    w_load = w_slv.subs(x, x - mu).subs(y, y - nu).doit()
    w_load = integrate(w_load, (mu, -b, b), (nu, -a, a)).doit()
    w_load = w_load.collect(I).rewrite(sin)
    if (lm == 0 and eta != 0):
        w_load = re((w_load.collect(1 / eta)) * conj).simplify()
    elif (eta == 0 and lm != 0):
        w_load = re((w_load.collect(1 / eta)) * conj).simplify()
    elif (eta == 0 and lm == 0):
        w_load = re(w_load * conj).simplify()
    else:
        w_load = re((w_load.collect(1 / eta).collect(1 / lm) * conj)).simplify()
    pprint('w(x, y, t) solution for rectangle load [2*a;2*b]')
    pprint(w_load)
    print("A coeff")
    pprint(Aa)
    print("B coeff")
    pprint(Bb)
    return w_load, A, B
    # if 'lm' in specs:
    #     lm = specs['lm']
    # else:
    #     lm = Symbol('lambda', real=True, nonzero=True)
    # if 'eta' in specs:
    #     eta = specs['eta']
    # else:
    #     eta = Symbol('eta', real=True, nonzero=True)
    # if lm == 0 and eta == 0:
    #     return 0, 0, 0
    # phi = lm * (v * t + xi - x) + eta * (chi - y)
    # delta = exp(I * phi)
    # Phi = Phile * cosh((H + z) * sqrt(lm ** 2 + eta ** 2)) * delta
    # dPhi_dz = diff(Phi, z)
    # wle = Symbol('omega_le')
    # w = wle * delta
    # border_eq = diff(w, t) - dPhi_dz.subs(z, 0)
    # Phi_le_solution = solve(border_eq, Phile)
    # try:
    #     Phi_le = Phi_le_solution[0]
    # except IndexError:
    #     return 0, 0, 0
    # print('hello')
    # Phi = simplify(Phi_le * cosh((H + z) * sqrt(lm ** 2 + eta ** 2)).subs(z, 0) * delta).doit()
    # dPhi_dt = diff(Phi, t)
    # model = D * (nabla4(w) + tf * diff(nabla4(w), t)) + pw * g * w + pi * h * diff(w, t,
    #                                                                                2) + pw * dPhidt + P * delta
    # wle = solve(model, wle)[0]
    #
    # K = tanh(H * sqrt(lm ** 2 + eta ** 2)) * sqrt(lm ** 2 + eta ** 2)
    # K.refine(Q.nonzero(K))
    # w = wle * delta
    # numer, denom = fraction(w)
    # numer = simplify(expand(P * numer, complex=True) / K)
    # denom = simplify(collect(simplify(denom), I).doit() / K)
    # A = collect(re(denom), D)
    # coefD = A.coeff(D)
    # A = (A - D * coefD) + D * factor(coefD)
    # B = factor(im(denom))
    #
    # dDelta_dxi = integrate(numer, xi, conds='none')
    # dDelta_dxi_dchi = integrate(dDelta_dxi, chi, conds='none')
    # numer = Subs(dDelta_dxi_dchi, xi, b).doit() - Subs(dDelta_dxi_dchi, xi, -b).doit()
    # numer = Subs(numer, chi, a).doit() - Subs(numer, chi, -a).doit()
    # Aa = Symbol('A', real=True, nonzero=True)
    # Bb = Symbol('B', real=True, nonzero=True)
    # conj = Aa - I * Bb
    # denom = simplify((Aa + I * Bb) * conj)
    # if lm == 0:
    #     numer = simplify(re(expand(numer * conj, complex=True)))
    #     return simplify(numer / denom), A, B
    # numer = - simplify(re(expand(numer * conj, complex=True)) * -1)
    # pprint(A)
    #
    # pprint(B)
    # return simplify(numer / denom), A, B
