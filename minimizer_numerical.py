import numpy as np
import matplotlib.pyplot as plt

def compute_nu(df_du_val, df_dv_val):
    num = np.cross(df_du_val, df_dv_val)
    denom = np.linalg.norm(num)
    nu = num/denom
    return nu

def compute_gram(df_du_val, df_dv_val):
    cp = np.cross(df_du_val, df_dv_val)
    cp_norm = np.linalg.norm(cp)
    g = cp_norm**2
    return g

def compute_gij(df_du_val, df_dv_val, i, j):
    # if i = 1, j = 1 => df_du.dot(df_du)
    # if i = 1, j = 2 => df_du.dot(df_dv) etc.
    result = None
    if i == 1:
        if j == 1:
            result = df_du_val.dot(df_du_val)
        else:
            result = df_du_val.dot(df_dv_val)
    else:
        if j == 1:
            result = df_dv_val.dot(df_du_val)
        else:
            result = df_dv_val.dot(df_dv_val)
    
    if result is None:
        raise ValueError("Error in compute_gij: Invalid number passed")

    return result

def compute_hij(nu_value, df_du2, df_dv2, df_dudv, i, j):
    result = None
    if i == 1:
        if j == 1:
            result = nu_value.dot(df_du2)
        else:
            result = nu_value.dot(df_dudv)
    else:
        if j == 1:
            result = nu_value.dot(df_dudv)
        else:
            result = nu_value.dot(df_dv2)

    if result is None:
        raise ValueError("Error in compute_hij: Invalid number passed")

    return result

def plot_surface(U, V, f_mesh):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(U, V, f_mesh[:,:,2], cmap='viridis')
    return fig
    # plt.show()


def main():
    N = 25
    u_range = np.linspace(1e-6, 2*np.pi, N)
    v_range = np.linspace(1e-6, 2*np.pi, N)

    du = u_range[1] - u_range[0]
    dv = v_range[1] - v_range[0]
    U, V = np.meshgrid(u_range, v_range, indexing='ij')
    f_mesh = np.zeros((N,N,3))
    H_mesh = np.zeros((N,N))
    nu_mesh = np.zeros((N,N,3))

    for i in range(N):
        for j in range(N):
            x_val = u_range[i]
            y_val = v_range[j]
            z_val = np.sin(np.sqrt(u_range[i]**2 + v_range[j]**2))
            f_mesh[i][j] = np.array([x_val, y_val, z_val]).astype(float).flatten()


    # plot my original surface
    original_fig = plot_surface(U, V, f_mesh)

    tol_eps = 1e-6
    eps = (min(du, dv)**2)/4

    not_minimal = True
    counter = 0

    while not_minimal:
        print("At iter = ", counter)

        # compute df_du and df_dv for our current f_mesh
        df_du = np.gradient(f_mesh, du, axis=0)
        df_dv = np.gradient(f_mesh, dv, axis=1)
        df_du2 = np.gradient(df_du, du, axis=0)
        df_dv2 = np.gradient(df_dv, dv, axis=1)
        df_dudv = np.gradient(df_du, dv, axis=1)

        # run our normal variation iteration
        # compute H at each point on our current surface
        not_minimal_iter = False
        H_val_total = 0
        # NOTE: fixed at the boundary by requirements of minimal surfaces
        # TODO: is this true
        for i in range(1, N-1):
            for j in range(1, N-1):
                surface_value = f_mesh[i][j]
                surface_x = surface_value[0]
                surface_y = surface_value[1]
                surface_z = surface_value[2]

                df_du_val = df_du[i][j]
                df_dv_val = df_dv[i][j]
                df_du2_val = df_du2[i][j]
                df_dv2_val = df_dv2[i][j]
                df_dudv_val = df_dudv[i][j]

                # compute nu at this point
                nu_val = compute_nu(df_du_val, df_dv_val)
                nu_mesh[i][j] = nu_val

                # compute H at this point
                h11 = compute_hij(nu_val, df_du2_val, df_dv2_val, df_dudv_val, 1, 1)
                h22 = compute_hij(nu_val, df_du2_val, df_dv2_val, df_dudv_val, 2,2)
                h12 = compute_hij(nu_val, df_du2_val, df_dv2_val, df_dudv_val, 1,2)
                g11 = compute_gij(df_du_val, df_dv_val, 1,1)
                g22 = compute_gij(df_du_val, df_dv_val, 2,2)
                g12 = compute_gij(df_du_val, df_dv_val, 1,2)
                g21 = compute_gij(df_du_val, df_dv_val, 2,1)

                gij = np.array([[g11, g12], [g21,g22]])

                # compute H
                det_gij = np.linalg.det(gij)
                prod = h11*g22 - 2*h12*g12 + h22*g11
                H_val = (1/(2*det_gij))*prod
                H_mesh[i][j] = H_val
                H_val_total += H_val**2


                if abs(H_val) > tol_eps:
                    # update my H_mesh because we need to do another iter
                    not_minimal_iter = True # if we are minimal, will never hit this line

                if not_minimal_iter == False:
                    not_minimal = False # break out
                else:
                    not_minimal = True

        # we are done with sweeping through this mesh

        # if we are still not minimal, then we need to update the mesh
        if not_minimal:
            for i in range(N):
                for j in range(N):
                    H_ij = H_mesh[i][j]
                    if abs(H_ij) > tol_eps:
                        new_val = f_mesh[i][j] + eps*H_mesh[i][j]*nu_mesh[i][j]
                        f_mesh[i][j] = new_val
                    else:
                        continue


        counter += 1
        print(H_val_total)


    # plot my new surface
    new_fig = plot_surface(U, V, f_mesh)
    original_fig.show()
    new_fig.show()
    plt.show()



if __name__ == '__main__':
    main()
