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
w_xy = Function('w')(x, y)
w_t = Function('w')(t)
w = Function('w')(x, y, t)
Phi_xyz = Function('Phi')(x, y, z)
Phi_t = Function('Phi')(t)
Phi = Function('Phi')(x, y, z, t)

w_le = Function('w')(lm, eta)
Phi_le = Function('Phi')(lm, eta)
w_let = Function('w')(lm, eta, t)
Phi_let = Function('Phi')(lm, eta, t)


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
    # model = model.subs(w, w_xy + w_t).subs(Phi, Phi_xyz + Phi_t).doit()
    # model = model.subs(w2, 0).subs(Phi2, 0).doit()
    # Phi_xyz = Function('Phi')(x, y, z)
    # Phi_t = Function('Phi')(t)
    # w_xy = Function('w')(x, y)
    # w_t = Function('w')(t)
    # model = model.replace(w1, w_xy).replace(Phi1, Phi_xyz).doit()
    # model = model.replace(w2, w_t).replace(Phi2, Phi_t).doit()
    print("Model with x := x - v*t substitution")
    pprint(model)

    # Fourier expressions
    w_f = (w_le + w_t) * delta
    k = Symbol('k')
    Phi_f = (Phi_le + Phi_t) * cosh((H + z) * k) * delta
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
    # iw_line = iw_line.subs(w, w_xy + w_t).subs(Phi, Phi_xyz + Phi_t).doit()
    # print('w=w1+w2, Phi = Phi1 + Phi2')
    # pprint(iw_line)
    # iw_line = iw_line.subs(w2, w_t * delta).subs(Phi2, Phi_t * delta).doit()
    # print('w2 = w(t)')
    # pprint(iw_line)
    # iw_line = iw_line.replace(w_xy, w_le).replace(Phi_xyz, Phi_le).doit()
    # print('w1 = w(l,e)')
    # pprint(iw_line)
    # iw_line_f = iw_line.subs(Phi_xyz, Phi_f).doit().subs(w_xy, w_f).doit()
    iw_line_f = iw_line.subs(Phi, Phi_f).doit().subs(w, w_f).subs(z, 0).doit()
    print("Ice-water border after subs")
    pprint(iw_line_f)
    Phi_le_slv = solve(iw_line_f, Phi_le)[0]
    pprint('Phi(lambda, eta, t) solution')
    pprint(Phi_le_slv)
    Phi_f_slv = Phi_f.subs(Phi_le, Phi_le_slv).doit().subs(z, 0).simplify()
    pprint('Phi(x, y, z) solution')
    pprint(Phi_f_slv)
    model_f = model.subs(Phi, Phi_f_slv).subs(w, w_f).subs(P, P * delta).doit()
    w_le_slv = solve(model_f, w_le)[0].doit()
    w_f_slv = w_f.subs(w_le, w_le_slv).doit()
    K = tanh(H * k_slv) * k_slv
    K_sym = Symbol('K')
    pprint('w(x, y, t) solution')
    w_le_slv = w_le_slv.rewrite(tanh).simplify(ratio=oo).subs(K, K_sym).doit()
    pprint(w_le_slv)
    w_t_lap_img = Symbol('W')
    w_le_slv_lap = w_le_slv.subs(diff(w_t, t, 2), tau ** 2 * w_t_lap_img).subs(diff(w_t, t), tau * w_t_lap_img).subs(
        w_t, w_t_lap_img).doit()
    w_t_lap_slv = solve(w_le_slv_lap, w_t_lap_img)[0].expand().collect(tau ** 2).collect(tau)
    print('w_lap solution')
    pprint(w_t_lap_slv)
    num, den = fraction(w_t_lap_slv)
    num /= K_sym
    den /= K_sym
    P_lap = laplace_transform(P * exp(I * freq * t) * Heaviside(t), t, tau)[0]
    num = num.subs(P, P_lap).doit()
    print('P laplace')
    pprint(num)
    den = den.expand().collect(tau ** 2).collect(tau)
    print('denom')
    pprint(den)
    k0_sym, k1_sym, k2_sym, n_sym = symbols('k0 k1 k2 n', positive=True)
    k2 = den.coeff(tau ** 2)
    k1 = den.coeff(tau)
    k0 = (den - tau * k1 - tau ** 2 * k2).simplify()
    den = den.subs(k2, k2_sym).subs(k1, k1_sym).subs(k0, k0_sym).doit()
    k1 = k1.collect(I).collect(K).collect(D * tf)
    D_tf_coeff = k1.coeff(D * tf)
    D_tf_coeff = D_tf_coeff.factor()
    k1 = (k1 - k1.coeff(D * tf) * D * tf) + D_tf_coeff * D * tf
    k0 = k0.expand().collect(I).collect(D)
    D_coeff = k0.coeff(D)
    D_coeff = D_coeff.factor()
    k0 = (k0 - k0.coeff(D) * D) + D_coeff * D
    I_coeff = k0.coeff(I)
    I_coeff = (I_coeff.collect(D * tf)).factor()
    k0 = (k0 - k0.coeff(I) * I) + I_coeff * I
    n = 1 / k2
    print('k0')
    pprint(k0)
    print('k1')
    pprint(k1)
    print('k2')
    pprint(k2)
    # pprint(k1)

    k0 = (den - tau * k1 - tau ** 2 * k2).simplify()
    den /= k2_sym
    l, m = symbols('l m', positive=True)
    den = den.expand().subs(k1_sym / k2_sym, l).subs(k0_sym / k2_sym, m).doit()
    num *= n_sym
    print('numer after coeff subs')
    pprint(num)
    print('denom after coeff subs')
    pprint(den)
    den_root_1, den_root_2 = solve(den, tau)
    pprint('den roots')
    pprint(den_root_1)
    pprint(den_root_2)

    r1, r2 = symbols('r1 r2', positive=True)
    den = (tau - r1) * (tau - r2)
    w_t_slv = inverse_laplace_transform(num / den, tau, t)
    p = I * freq
    p_sym = Symbol('p', positive=True)
    w_t_slv = w_t_slv.subs(p, p_sym).doit().combsimp()
    w_t_slv = w_t_slv.rewrite(exp).factor().simplify().subs(p, p_sym).doit()
    pprint(w_t_slv)
    # w_le_slv = w_le_slv.subs(diff(w_t, t), 0).subs(w_t, w_t_slv).doit()
    dw_t = diff(w_t_slv, t).doit()
    print('diff w_t, t')
    pprint(dw_t)
    d2w_t2 = diff(w_t_slv, t, 2).doit()
    print('diff w_t, t, 2')
    pprint(d2w_t2)
    w_t_sym = Symbol('W0')
    dw_t_sym = Symbol('W1')
    d2w_t2_sym = Symbol('W2')
    w_le_slv = w_le_slv.subs(diff(w_t, t, 2), d2w_t2_sym).doit().subs(diff(w_t, t), dw_t_sym).subs(w_t, w_t_sym).doit()
    w_le_slv = w_le_slv.cancel().simplify(ratio=oo).collect(w_t_sym).collect(dw_t_sym).collect(d2w_t2_sym)
    print('w_le after W_i subs')
    pprint(w_le_slv)
    w_le_num, w_le_den = fraction(w_le_slv)
    w0_coeff = w_le_num.coeff(w_t_sym)
    w1_coeff = w_le_num.coeff(dw_t_sym)
    w2_coeff = w_le_num.coeff(d2w_t2_sym)
    w_le_num_remain = (w_le_num - w0_coeff * w_t_sym - w1_coeff * dw_t_sym - w2_coeff * d2w_t2_sym).simplify()
    pprint('num_remain')
    pprint(w_le_num_remain)
    print('W0_coeff / w_le_den')
    pprint((w0_coeff.factor(deep=True) / w_le_den).cancel())
    print('W1_coeff')
    pprint(
        w1_coeff.factor(deep=True).collect(I).collect(D).collect(tf).collect(K_sym).collect(tf).collect(
            lm ** 2 * v ** 2))
    print('W2_coeff')
    pprint(w2_coeff.factor(deep=True).collect(I).collect(D))
    print('denom')
    pprint(
        w_le_den.factor(deep=True).collect(I).collect(D).collect(K_sym).collect(tf).collect(lm ** 2 * v ** 2))
    exit(0)
    # return w_t_slv.simplify(ratio=oo)
