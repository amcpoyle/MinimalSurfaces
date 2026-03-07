import numpy as np
import sympy as sp
import casadi as ca
import matplotlib.pyplot as plt


# symbolic
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
    # g = gram determinant = det(gij)
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

# f = f + H*nu
# has to vanish on the boundary - ignore?
# do not iterate over the boundary?
# take to 0 near boundary manually: f = f + H*nu(u-u0)(u1-u)(v-v0)(v1-v)
# where U = [u0, u1] x [v0,v1]
# mean curvature flow implementation


# numeric
def compute_gij(u_val, v_val, i, j):
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

def compute_H_numeric(u_val, v_val):
    h11 = compute_hij_numeric(f, v_, 1, 1)
    h12 = compute_hij_numeric(f, v_, 1, 2)
    h22 = compute_hij_numeric(f, v_, 2, 2)

    g22 = compute_gij_numeric(f, v_, 2, 2)
    g12 = compute_gij_numeric(f, v_, 1, 2)
    g11 = compute_gij_numeric(f, v_, 1, 1)

    g = compute_gram_numeric(f, v_)

    H = (1/(2*g))*(h11*g22 - 2*h12*g12 + h22*g11)
    return H

def minimize_surface():
    pass

def main():
    # define our surface
    N = 5
    u_range = np.linspace(1e-6, 2*np.pi, N)
    v_range = np.linspace(1e-6, 2*np.pi, N)

    du = u_range[1] - u_range[0]
    dv = v_range[1] - v_range[0]
    U, V = np.meshgrid(u_range, v_range, indexing='ij')

    u, v = sp.symbols('u v')
    f = sp.Matrix([u, v, sp.sin(sp.sqrt(u**2 + v**2))])
    nu = compute_nu(f, [u,v])

    # calculate f at every point in the mesh
    f_mesh = np.zeros((N,N, 3))
    nu_mesh = np.zeros((N,N, 3))
    phi_mesh = np.zeros((N,N, 3))

    # test = compute_nu(f, [u,v])
    # print(test.subs({(u, np.pi/4), (v, np.pi/2)}))
    
    # initial population of f_mesh
    for i in range(N):
        for j in range(N):
            result = f.subs({(u, u_range[i]), (v, v_range[j])})
            f_mesh[i][j] = np.array(result).astype(float).flatten()


    # initial population of nu_mesh
    for i in range(N):
        for j in range(N):
            result = nu.subs({(u, u_range[i]), (v, v_range[j])})
            nu_mesh[i][j] = np.array(result).astype(float).flatten()


    iter_counter = 0
    tol_eps = 1e-3
    current_H_total = 100 # just a starting point
    current_phi = 1 # init value, f_eps = f
    eps = 1e-3

    while current_H_total > tol_eps:
        print("iteration: ", iter_counter)

        integral = 0
        for i in range(N):
            for j in range(N):
                u, v = sp.symbols('u v')
                perturbation = sp.Matrix((eps * current_phi * nu_mesh[i][j]).tolist())
                f_eps = f + perturbation

                # compute H
                H = compute_H(f_eps, [u,v])
                H_val = H.subs({(u, u_range[i]), (v, v_range[j])}).evalf()

                # compute gram
                g = compute_gram(f_eps, [u,v])
                g_val = g.subs({(u, u_range[i]), (v, v_range[j])}).evalf()

                integrand = current_phi*2*float(H_val)*np.sqrt(float(g_val))*du*dv
                print(integrand)
                integral += integrand
        
        current_H_total = abs(integral)
        print("Current value: ", current_H_total)
        current_phi = H_val
        iter_counter += 1







if __name__ == '__main__':
    main()
