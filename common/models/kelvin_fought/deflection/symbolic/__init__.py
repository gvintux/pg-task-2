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
phi = Symbol('phi', positive=True)
Pmax = Symbol('P_max', positive=True)
Pmin = Symbol('P_min', positive=True)
Pmin = 0
# phi = 0
p = freq * I
p_sym = Symbol('p', positive=True)


def nabla4(func):
    d4_dx4 = diff(func, x, 4)
    d4_dy4 = diff(func, y, 4)
    d4_dx2_dy2 = diff(diff(func, x, 2), y, 2)
    return d4_dx4 + 2 * d4_dx2_dy2 + d4_dy4


def fourier_integral(func, dlt):
    return 1 / (2 * PI) * Integral(func * dlt, (lm, -oo, +oo), (eta, -oo, +oo))


def deflection_solve(**specs):
    init_printing(num_columns=120)
    if 'lm' in specs:
        global lm
        lm = 0
    if 'eta' in specs:
        global eta
        eta = 0
    delta = exp(-I * (lm * (x - v * t) + eta * y))
    global P
    model = Eq(D * (nabla4(w) + tf * diff(nabla4(w), t)) + pw * g * w + pi * h * diff(w, t, 2) + pw * diff(Phi, t), P)
    print("General model")
    pprint(model)
    u = Symbol('u')
    ss = u - v * t
    model = model.subs(x, ss).doit()
    model = model.replace(ss, x).doit()
    print("Model with x := x - v*t substitution")
    pprint(model)

    # Fourier expressions
    w_f = (w_le + w_t) * delta
    k = Symbol('k')
    Phi_f = (Phi_le + Phi_t) * cosh((H + z) * k) * delta
    laplace_rule = Eq(diff(Phi_f, x, 2) + diff(Phi_f, y, 2) + diff(Phi_f, z, 2), 0)
    pprint("Laplace rule")
    pprint(laplace_rule)
    k_slv = solve(laplace_rule, k)
    print("k solutions")
    pprint(k_slv)
    k_slv = k_slv[2]
    Phi_f = Phi_f.subs(k, k_slv).doit()
    K = tanh(H * k_slv) * k_slv
    # means K
    K_sym = Symbol('K', real=True)
    # ice-water line z = 0
    iw_line = Eq(diff(w, t).doit(), diff(Phi, z))
    print('Ice-water border equation')
    pprint(iw_line)
    iw_line = iw_line.subs(x, ss).doit()
    iw_line = iw_line.replace(ss, x).doit()
    print('Ice-water border equation with x := x - v*t substitution')
    pprint(iw_line)
    iw_line_f = iw_line.subs(Phi, Phi_f).doit().subs(w, w_f).subs(z, 0).doit()
    print("Ice-water Fourier")
    pprint(iw_line_f)
    Phi_le_slv = solve(iw_line_f, Phi_le)[0]
    pprint('Phi(l, e, t) solution')
    pprint(Phi_le_slv)
    Phi_f_slv = Phi_f.subs(Phi_le, Phi_le_slv).doit().subs(z, 0).simplify()
    pprint('Phi(x, y, t) solution')
    pprint(Phi_f_slv)
    model_f = model.subs(Phi, Phi_f_slv).subs(w, w_f).subs(P, P * delta).doit()
    w_le_slv = solve(model_f, w_le)[0].doit()
    w_le_slv = w_le_slv.rewrite(tanh).simplify(ratio=oo).subs(K, K_sym).doit()
    # pprint('w(l, e) solution')
    # pprint(w_le_slv)
    w_f_slv = w_f.subs(w_le, w_le_slv).doit()
    pprint('w(l, e, t) solution')
    pprint(w_f_slv)
    w_t_lap_img = Function('W')(tau)
    w_f_slv_lap = w_f_slv.subs(diff(w_t, t, 2), tau ** 2 * w_t_lap_img).subs(diff(w_t, t), tau * w_t_lap_img).subs(
        w_t, w_t_lap_img).doit()
    w_tau = solve(w_f_slv_lap, w_t_lap_img)[0].expand().collect(tau ** 2).collect(tau)
    print('w(tau) solution')
    pprint(w_tau)
    w_tau_num, w_tau_den = fraction(w_tau)
    w_tau_num /= K_sym
    w_tau_den /= K_sym
    w_tau_den = w_tau_den.expand(deep=True).collect([tau ** 2, tau])
    print('w(tau) num')
    pprint(w_tau_num)
    print('w(tau) den')
    pprint(w_tau_den)
    # phi_sym = Symbol('phi', positive=True)
    # phase = Symbol('phi0', positive=True)
    phase = 1
    P = exp(I * (freq * t)) * phase * Heaviside(t) + 1
    P *= (Pmax - Pmin) / 2
    P += Pmin
    pprint(P)
    P_tau = laplace_transform(P, t, tau)[0]
    pprint(P_tau)
    P_tau = P_tau.expand()
    P_tau_a = P_tau.args[0].factor(deep=True).subs(p, p_sym).doit()
    P_tau_b = P_tau.args[1].factor(deep=True).subs(p, p_sym).doit()
    # P_tau = P_tau_a + P_tau_b
    print('P_tau_a')
    pprint(P_tau_a)
    print('P_tau_b')
    pprint(P_tau_b)
    k0_sym, k1_sym, k2_sym, n_sym, l_sym, m_sym = symbols('k0 k1 k2 n l m', positive=True)
    k2 = w_tau_den.coeff(tau ** 2)
    k1 = w_tau_den.coeff(tau)
    k0 = (w_tau_den - tau * k1 - tau ** 2 * k2).simplify()
    w_tau_den = w_tau_den.subs(k2, k2_sym).subs(k1, k1_sym).subs(k0, k0_sym).doit()
    print('w_tau_den')
    pprint(w_tau_den)
    k1 = k1.collect([D * tf, I * lm * v])
    k1 = k1.args[0].factor(deep=True) + k1.args[1]
    pprint('k0')
    pprint(k0)
    pprint('k1')
    pprint(k1)
    pprint('k2')
    pprint(k2)
    n = 1 / k2
    P_tau_a *= n_sym
    P_tau_b *= n_sym
    w_tau_den = w_tau_den.subs(k1_sym, l_sym).subs(k0_sym, m_sym).subs(k2_sym, 1).doit()
    print('w_tau_den reduced')
    pprint(w_tau_den)
    w_tau_den_r1, w_tau_den_r2 = solve(w_tau_den, tau)
    print('w_tau_den roots')
    pprint([w_tau_den_r1, w_tau_den_r2])
    w_tau_den = (tau - w_tau_den_r1) * (tau - w_tau_den_r2)
    print('w_tau_den decomposition')
    pprint(w_tau_den)
    w_t_slv_a = inverse_laplace_transform(P_tau_a / w_tau_den, tau, t)
    print('w(t)_a solution ')
    pprint(w_t_slv_a)
    w_t_slv_b = inverse_laplace_transform(P_tau_b / w_tau_den, tau, t)
    print('w(t)_b solution')
    pprint(w_t_slv_b)
    dw_t_a = diff(w_t_slv_a, t).factor(deep=True).powsimp()
    print('dw(t)_a')
    pprint(dw_t_a)
    dw_t_b = diff(w_t_slv_b, t).factor(deep=True).powsimp()
    print('dw(t)_b')
    pprint(dw_t_b)
    d2w_t2_a = diff(dw_t_a, t).factor(deep=True).powsimp()
    print('d2w(t)_a')
    pprint(d2w_t2_a)
    d2w_t2_b = diff(dw_t_b, t).factor(deep=True).powsimp()
    print('d2w(t)_b')
    pprint(d2w_t2_b)
    W0_sym, W1_sym, W2_sym = symbols('W0 W1 W2')
    w_f_slv = w_f_slv.subs(diff(w_t, t, 2), W2_sym).subs(diff(w_t, t), W1_sym).subs(w_t, W0_sym).doit()
    print('w_f_slv after dt subs')
    w_f_slv /= delta
    w_f_slv = w_f_slv.factor(deep=True)
    pprint(w_f_slv)
    w_f_slv_num, w_f_slv_den = fraction(w_f_slv)
    w_f_slv_num /= K_sym
    w_f_slv_den /= K_sym
    w_f_slv_num = w_f_slv_num.expand(deep=True).collect([W2_sym, W1_sym, W0_sym])
    w_f_slv_den = w_f_slv_den.expand(deep=True)

    # w_f_slv_num = w_f_slv_num.collect([W2_sym, W1_sym, W0_sym]).subs(k1, k1_sym).subs(k2, k2_sym).doit()
    # w_f_slv_den = w_f_slv_den.collect([I, D*tf, W0_sym])
    print('w_f_slv_num')
    pprint(w_f_slv_num)
    w_f_slv_num_W1_coeff = w_f_slv_num.coeff(W1_sym)
    w_f_slv_num_W2_coeff = w_f_slv_num.coeff(W2_sym)

    print('w_f_slv_num')
    pprint(w_f_slv_num)

    w_f_slv_den = w_f_slv_den.collect([I, D])
    w_f_slv_den_re = re(w_f_slv_den).collect([lm ** 2 * v ** 2]).powsimp(deep=True)
    w_f_slv_den_im = im(w_f_slv_den).collect([D, tf, lm * v]).powsimp(deep=True)
    print('w_f_slv_den_re')
    pprint(w_f_slv_den_re)
    print('w_f_slv_den_im')
    pprint(w_f_slv_den_im)

    print('w_f_slv_den')
    pprint(w_f_slv_den)
    exit(0)

    num_a, num_b = num.args[0].simplify(), num.args[1].simplify()
    print('P laplace')
    pprint(num)
    print('denom')
    pprint(den)
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
    den = den.expand().subs(k1_sym / k2_sym, l).subs(k0_sym / k2_sym, m).subs(p, p_sym).doit()
    num *= n_sym
    print('numer after coeff subs')
    pprint(num)
    print('denom after coeff subs')
    pprint(den)
    den_root_1, den_root_2 = solve(den, tau)
    pprint('den roots')
    pprint(den_root_1)
    pprint(den_root_2)

    r1, r2, r = symbols('r1 r2 r', positive=True)
    den = tau * (tau + r)
    num = num.subs(p, p_sym).doit()
    num_a, num_b = num.factor(deep=True).expand().args[0].factor(deep=True).collect([tau ** 2, tau]),
    num.factor(deep=True).expand().args[1].factor(deep=True).collect([tau ** 2, tau])
    print('P Laplace')

    w_t_slv_a = inverse_laplace_transform(num_a / den, tau, t)
    pprint(fraction(w_t_slv_a)[0].expand(deep=True).cancel().factor(deep=True))
    w_t_slv_b = inverse_laplace_transform(num_b / den, tau, t)
    pprint(fraction(w_t_slv_b)[0].expand(deep=True).cancel().factor(deep=True))
    # pprint()
    # exit(0)
    # w_t_slv_b = 0
    # w_t_slv_a = w_t_slv_a.subs(p, p_sym).doit().combsimp()
    # w_t_slv_b = w_t_slv_b.subs(p, p_sym).doit().combsimp()
    # w_t_slv_a = w_t_slv_a.rewrite(exp).factor().simplify().subs(p, p_sym).collect(Pmax).collect(Pmin).doit()
    # w_t_slv_b = w_t_slv_b.rewrite(exp).factor().simplify().subs(p, p_sym).collect(Pmax).collect(Pmin).doit()
    pprint('w_t_slv_a')
    pprint(w_t_slv_a.collect([exp(r * t), exp(p_sym * t)]))
    pprint('w_t_slv_b')
    pprint(w_t_slv_b.collect([exp(r * t), exp(p_sym * t)]))
    w_t_slv = (w_t_slv_a + w_t_slv_b) / den
    w_t_slv_num, w_t_slv_den = fraction(w_t_slv)
    w_t_slv_num = (w_t_slv_num).collect(
        [exp(r * t), exp(p_sym * t), p_sym, r, (r - p)])
    pprint(w_t_slv_num)
    # exit(0)
    print('W0')
    pprint((w_t_slv_num / w_t_slv_den).simplify())
    print('W1')
    pprint((diff(w_t_slv_num, t, 1) / w_t_slv_den).simplify(ratio=oo))
    print('W2')
    pprint((diff(w_t_slv_num, t, 2) / w_t_slv_den).simplify(ratio=oo))
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
    w_le_num = ((w_le_num / K_sym).expand(deep=True).cancel().collect([w_t_sym, dw_t_sym, d2w_t2_sym]))
    w_le_den = ((w_le_den / K_sym).expand(deep=True).cancel())

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
    print('w_le denom')
    pprint(
        w_le_den.factor(deep=True).collect(I).collect(D).collect(K_sym).collect(tf).collect(lm ** 2 * v ** 2))
    exit(0)
    # return w_t_slv.simplify(ratio=oo)
