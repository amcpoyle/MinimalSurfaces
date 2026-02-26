import numpy as np
import sympy as sp
import casadi as ca
import matplotlib.pyplot as plt

def compute_gij(f, v_, i, j):
    # v = vars
    u, v = v_

    df_ui = None
    df_uj = None
    if i == 1:
        df_ui = f.diff(u)
    elif i == 2:
        df_ui = f.diff(v)
    else:
        raise ValueError("Error in compute_gij: invalid number 'i'")

    if j == 1:
        df_uj = f.diff(u)
    elif j == 2:
        df_uj = f.diff(v)
    else:
        raise ValueError("Error in compute_gij: invalid number 'j'")

    # compute the inner product
    gij = df_ui.dot(df_uj)
    return gij

def compute_hij(f, v_, i, j):
    # compute the norm nu first
    u, v = v_
    nu = compute_nu(f, v_)

    dnu_ui = None
    df_uj = None

    if i == 1:
        dnu_ui = nu.diff(u)
    elif i == 2:
        dnu_ui = nu.diff(v)
    else:
        raise ValueError("Error in compute_hij: invalid number 'i'")

    if j == 1:
        df_uj = f.diff(u)
    elif j == 2:
        df_uj = f.diff(v)
    else:
        raise ValueError("Error in compute_hij: invalid number 'j'")

    hij = -(dnu_ui.dot(df_uj))
    return hij


def compute_nu(f, v_):
    u, v = v_

    df_du = f.diff(u)
    df_dv = f.diff(v)

    numerator = df_du.cross(df_dv)
    denom = numerator.norm()
    nu = numerator/denom
    return nu

def compute_gram(f, v_):
    # g = gram determinant
    u, v = v_
    df_du = f.diff(u)
    df_dv = f.diff(v)
    cross_product = df_du.cross(df_dv)
    cp_norm = cross_product.norm()
    g = cp_norm**2
    return g

def compute_H(f, v_):
    h11 = compute_hij(f, v_, 1, 1)
    h12 = compute_hij(f, v_, 1, 2)
    h22 = compute_hij(f, v_, 2, 2)

    g22 = compute_gij(f, v_, 2, 2)
    g12 = compute_gij(f, v_, 1, 2)
    g11 = compute_gij(f, v_, 1, 1)

    g = compute_gram(f, v_)

    H = (1/(2*g))*(h11*g22 - 2*h12*g12 + h22*g11)
    return H

def plot_surface(f, v_, U, V):
    # plots the surface element f
    f_func = sp.lambdify(v_, f, 'numpy')

    result = f_func(U, V)
    X, Y, Z = result[0].squeeze(), result[1].squeeze(), result[2].squeeze()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def compute_H_surface(f, v_):
    # goal: def of minimal surface - we need H = 0 for all (u,v)
    # => we need int int H^{2} sqrt(g) du dv = 0 for this to hold
    # this works because we are squaring H so neg and pos curvatures don't cancel
    u, v = v_
    
    u_range = np.linspace(1e-6, 2*np.pi, 250)
    v_range = np.linspace(1e-6, 2*np.pi, 250)
    
    H_sym = compute_H(f, [u,v])
    g_sym = sp.sqrt(compute_gram(f, [u,v]))

    integrand = (H_sym**2)*g_sym
    func = sp.lambdify([u,v], integrand, 'numpy')
    grid = func(U,V)

    total = np.trapz(np.trapz(grid, v_range, axis=1), u_range)
    return total

def minimize_surface():
    pass

def main():
    # define our surface
    u_range = np.linspace(1e-6, 2*np.pi, 250)
    v_range = np.linspace(1e-6, 2*np.pi, 250)
    U, V = np.meshgrid(u_range, v_range)

    u, v = sp.symbols('u v')
    f = sp.Matrix([u, v, sp.sin(sp.sqrt(u**2 + v**2))])

    # plot_surface(f, [u,v], U, V)


    # test = f.subs([(u, np.pi/4), (v, np.pi/3) ])


if __name__ == '__main__':
    main()
