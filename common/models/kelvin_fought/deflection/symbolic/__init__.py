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


# def diff_t(func):
#     return diff(func, t) - v * diff(func, x)
#
#
# def diff_t2(func):
#     return diff(func, t, 2) - 2 * v * diff(diff(func, x), t) + v ** 2 * diff(func, x, 2)

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
        Phi_f = Phi_f.subs(k, k_slv[1]).doit()
    if eta == 0 and lm != 0:
        Phi_f = Phi_f.subs(k, k_slv[0]).doit()
    if eta == 0 and lm == 0:
        Phi_f = Phi_f.subs(k, k_slv[0]).doit()
    if eta != 0 and lm != 0:
        Phi_f = Phi_f.subs(k, k_slv[2]).doit()
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
    pprint(iw_line)
    # iw_line = iw_line.replace(w1, w_le).replace(Phi1, Phi_le).doit()
    # print('w1 = w(l,e)')
    # pprint(iw_line)
    # iw_line_f = iw_line.subs(Phi1, Phi_f).doit().subs(w1, w_f).doit()
    iw_line_f = iw_line.subs(Phi, Phi_f).doit().subs(w, w_f).doit()

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
    w_let_slv = model.subs(Phi, Phi_f_slv).subs(w, w_f).subs(P, P * delta).doit()
    w_let_slv = solve(w_let_slv, w_let)[0].doit()
    w_f_slv = w_f.subs(w_let, w_let_slv).doit() * delta
    pprint('w(x, y, t) solution')
    pprint(w_let_slv)
    w0 = Symbol('w_0')
    w1 = Symbol('w_1')
    w2 = Symbol('w_2')
    w_f_slv = w_f_slv.replace(diff(w_let, t), tau * w1).replace(diff(w_let, t, 2), tau ** 2 * w2).replace(w_let,
                                                                                                          w0).doit()

    w_f_slv = w_f_slv / delta / delta
    num, den = fraction(w_f_slv)
    K = tanh(H * sqrt(lm ** 2 + eta ** 2)) * sqrt(lm ** 2 + eta ** 2)
    num = (num / K).simplify()
    den = (den / K).expand(deep=True).simplify()
    print('*' * 80)
    pprint(num.cancel().collect(tau ** 2).collect(tau))
    print('*' * 80)
    pprint(den.cancel().collect(tau ** 2).collect(tau))
    print('*' * 80)
    return None
    # numer, denom = fraction(w_let_slv)
    # K = sqrt(lm ** 2 + eta ** 2) * tanh(H * sqrt(lm ** 2 + eta ** 2))
    # pprint("COEFF")
    # pprint(K)
    # numer /= K
    # print("NUMER")
    # pprint(numer.expand(deep=True))
    # denom = (denom / K).expand(deep=True)
    # print("DENOM")
    # pprint(denom)

    P_lap = P * exp(I * freq * t) * Heaviside(t)
    P_lap = laplace_transform(P_lap, t, tau)[0].doit()
    pprint(P_lap)
    w_let_lap = Function('w')(lm, eta, tau)
    dw_dt_lap = tau * w_let_lap
    d2w_dt2_lap = tau ** 2 * w_let_lap
    # model = (model.subs(w, w_let_slv * delta).subs(Phi, (Phi_let_slv * delta).subs(z, 0)).doit() / delta).expand(deep=True)
    model = model.subs(P, P_lap).subs(diff(w_let, t), dw_dt_lap).subs(diff(w_let, t, 2), d2w_dt2_lap).doit().expand(
        deep=True)
    # sol = solve(model, tau)
    model = model.subs(w, w1 + w2).subs(Phi, Phi1 + Phi2).doit().expand(deep=True)
    model = model.subs(P, P_lap).subs(diff(w2, t), dw_dt_lap).subs(diff(w2, t, 2), d2w_dt2_lap).subs(Phi2,
                                                                                                     0).doit().expand(
        deep=True)
    model = model.subs(Phi1, Phi_let_slv.subs(Phi2, 0)).doit()
    model = Eq(model, P_lap)
    w_t_lap = solve(model, w2)[0]
    pprint(inverse_laplace_transform(P_lap, tau, t))
    pprint(inverse_laplace_transform(w_t_lap, tau, t))
    model = model.subs(w_t, w_t_lap).subs(w2, 0).doit()
    model = model.subs(w1, w_let_slv).subs(diff(w_let, t), 0).subs(diff(w_let, t, 2), 0).subs(P_lap, P).doit()

    pprint(model)

    return None
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
    pprint(w_le_slv_simp)
    w_slv = w_f.subs(w_le, w_le_slv_simp).doit()
    pprint('w(x, y, t) solution')
    pprint(w_slv)
    pprint(Aa)
    pprint(Bb)
    print("MODEL")
    k = sqrt(lm ** 2 + eta ** 2)
    wfun = Function('w')(lm, eta, t)
    MOD = D * k ** 4 * wfun + tf * D * (k ** 4 * I * lm * v * wfun + diff(wfun, t)) + pw * g * wfun + pi * h * (
        -lm ** 2 * v ** 2 * wfun + 2 * I * lm * v * diff(wfun, t) + diff(wfun, t, 2) - 2 * v * lm * v * wfun - I * diff(
            wfun, t) + v ** 2 * lm ** 2 * wfun) + pw * (
        lm * v / (k * tanh(H * k)) * (I * lm * v * wfun + diff(wfun, t)) + v ** 2 * lm ** 2 * I * wfun / (
            k * tanh(H * k)))
    MOD = MOD.simplify(ratio=oo).collect(D)
    MOD = Eq(MOD, P)
    MOD = MOD.subs(P, laplace_transform(P * exp(I * freq * t) * Heaviside(t), t, tau)[0]).doit()
    wfun_lap = Function('w')(lm, eta, tau)
    MOD = MOD.subs(diff(wfun, t), tau * wfun_lap).subs(diff(wfun, t, 2), tau ** 2 * wfun_lap).subs(wfun,
                                                                                                   wfun_lap).doit()
    MOD = MOD.expand()
    print("=" * 80)
    print("=" * 80)
    pprint(MOD.collect(tau))
    print("=" * 80)
    print("=" * 80)
    wfun_lap_slv = solve(MOD, wfun_lap)[0]
    wfls_n, wfls_d = fraction(wfun_lap_slv)
    wfls_n = (wfls_n / K).trigsimp().subs(k, Symbol('K')).doit()
    wfls_d = (wfls_d / K).trigsimp().subs(k, Symbol('K')).doit()
    print('NUMER')
    pprint(wfls_n)
    print('DENOM')
    wfls_d = (wfls_d.simplify(ratio=oo)).expand().collect(I).collect(D).collect(h).collect(tau).powsimp(deep=True)
    wfls_d_im = wfls_d.coeff(I)
    wfls_d_re = wfls_d - I * wfls_d_im

    print('DENOM re')
    pprint(wfls_d_re.subs(im(K), 0).subs(re(K), K).doit())
    print('DENOM im')
    pprint(wfls_d_im.subs(im(K), 0).subs(re(K), K).doit())

    # w_le_lap = laplace_transform(MOD, t, tau)
    # pprint(w_le_lap.doit())

    # # load size
    # w_load = w_slv.subs(x, x - mu).subs(y, y - nu).doit()
    # w_load = integrate(w_load, (mu, -b, b), (nu, -a, a)).doit()
    # w_load = w_load.collect(I).rewrite(sin)
    # if (lm == 0 and eta != 0):
    #     w_load = re((w_load.collect(1 / eta)) * conj).simplify()
    # elif (eta == 0 and lm != 0):
    #     w_load = re((w_load.collect(1 / eta)) * conj).simplify()
    # elif (eta == 0 and lm == 0):
    #     w_load = re(w_load * conj).simplify()
    # else:
    #     w_load = re((w_load.collect(1 / eta).collect(1 / lm) * conj)).simplify()
    # pprint('w(x, y, t) solution for rectangle load [2*a;2*b]')
    # pprint(w_load)
    # print("A coeff")
    # pprint(Aa)
    # print("B coeff")
    # pprint(Bb)
    # return w_load, A, B
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
