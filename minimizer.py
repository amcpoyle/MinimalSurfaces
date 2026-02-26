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

    if i == 1:
        df_uj = f.diff(u)
    elif i == 2:
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


def compute_H():
    pass

def minimize_surface():
    pass

def main():
    # define our surface
    u_range = np.linspace(0, 2*np.pi, 250)
    v_range = np.linspace(0, 2*np.pi, 250)

    u, v = sp.symbols('u v')
    f = sp.Matrix([3*u, 3*v, 3*sp.sin(sp.sqrt(u**2 + v**2))])

    # goal: minimize the surface area of f

    # test = f.subs([(u, np.pi/4), (v, np.pi/3) ])


if __name__ == '__main__':
    main()
