import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

"""
helper functions for computations
"""
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
    ax.plot_surface(f_mesh[:,:,0], f_mesh[:,:,1], f_mesh[:,:,2], cmap='viridis')
    return fig
    # plt.show()

"""
this is the main function that runs the numerical minimization
"""
def run_minimizer(N, u_range, v_range, du, dv, U, V, f_mesh, H_mesh, nu_mesh, tol_eps, eps):

    not_minimal = True
    counter = 0

    while not_minimal:
        print("At iter = ", counter)

        # compute df_du and df_dv for our current f_mesh
        # TODO: looks if that's the issue
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

                # if H_val is nan, break out


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
                for j in range(f_mesh.shape[1]):
                    H_ij = H_mesh[i][j]
                    if abs(H_ij) > tol_eps:
                        new_val = f_mesh[i][j] + eps*H_mesh[i][j]*nu_mesh[i][j]
                        f_mesh[i][j] = new_val
                    else:
                        continue


        counter += 1
        print(H_val_total)
        
    return f_mesh, H_mesh, nu_mesh

def run_minimizer_animation(N, u_range, v_range, du, dv, U, V, f_mesh, H_mesh, nu_mesh, tol_eps, eps, frame_freq, fix_u_boundary=True, fix_v_boundary=True, periodic=False, frac=0.01):

    not_minimal = True
    counter = 0
    plotly_frames = []
    det_gij_tracking = None
    had_singularity = False

    while not_minimal:
        det_gij_min = np.inf
        det_gij_min_idx = None

        print("At iter = ", counter)

        # add our mesh to plotly_frames
        if counter % frame_freq == 0:
            f_plot = np.concatenate([f_mesh, f_mesh[0:1]], axis=0) if periodic else f_mesh
            frame = go.Frame(
                    data = go.Surface(
                        x=f_plot[:,:,0],
                        y=f_plot[:,:,1],
                        z=f_plot[:,:,2],
                        colorscale='Viridis',
                        cmin=-1,cmax=1),
                    name = str(counter)
                )
            plotly_frames.append(frame)

        df_dv = np.gradient(f_mesh, dv, axis=1)
        df_dv2 = np.gradient(df_dv, dv, axis=1)

        if periodic:
            df_du = (np.roll(f_mesh, -1, axis=0) - np.roll(f_mesh, 1, axis=0)) / (2*du)
            df_du2 = (np.roll(f_mesh, -1, axis=0) - 2*f_mesh + np.roll(f_mesh, 1, axis=0)) / (du**2)
            df_dudv = (np.roll(df_dv, -1, axis=0) - np.roll(df_dv, 1, axis=0)) / (2*du)
        else:
            df_du = np.gradient(f_mesh, du, axis=0)
            df_du2 = np.gradient(df_du, du, axis=0)
            df_dudv = np.gradient(df_du, dv, axis=1)


        # run our normal variation iteration
        # compute H at each point on our current surface
        not_minimal_iter = False
        H_val_total = 0
        # NOTE: fixed at the boundary by requirements of minimal surfaces
        # TODO: is this true
        # TODO: neck pinch singularity don't fix?
        Nv = f_mesh.shape[1]
        i_range = range(N) if (periodic or not fix_u_boundary) else range(1, N-1)
        j_range = range(1, Nv-1) if fix_v_boundary else range(Nv)
        for i in i_range:
            for j in j_range:
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
                if det_gij > 0:
                    # det_gij_min = min(det_gij_min, det_gij) # maybe update with new min if we have it
                    if det_gij < det_gij_min:
                        # we have a new min
                        det_gij_min = det_gij
                        det_gij_min_idx = [i,j]

                    

                prod = h11*g22 - 2*h12*g12 + h22*g11
                H_val = (1/(2*det_gij))*prod # H --> infinity when det_gij --> 0
                H_mesh[i][j] = H_val

                if np.isnan(H_val):
                    continue # we are at a bad point (not singularity though)

                H_val_total += H_val**2

                if abs(H_val) > tol_eps:
                    # update my H_mesh because we need to do another iter
                    not_minimal_iter = True # if we are minimal, will never hit this line

                if not_minimal_iter == False:
                    not_minimal = False # break out
                else:
                    not_minimal = True

        # we are done with sweeping through this mesh
        # if eps_reduction:
        #     r = np.mean(np.linalg.norm(f_mesh, axis=2))
        #     if r < 0.9:
        #         not_minimal = False
        if det_gij_tracking is None and np.isfinite(det_gij_min):
            det_gij_tracking = det_gij_min

        if det_gij_tracking is not None and det_gij_min < frac * det_gij_tracking:
            print(f"Singularity at iter {counter}: min det(g) = {det_gij_min:.3e}")
            print(f"({100*det_gij_min/det_gij_tracking:.2f}% of initial). Stopping.")
            # singularity_pts = det_gij
            had_singularity = True
            not_minimal = False


        # if we are still not minimal, then we need to update the mesh
        if not_minimal:
            for i in range(N):
                for j in range(f_mesh.shape[1]):
                    H_ij = H_mesh[i][j]
                    if abs(H_ij) > tol_eps:
                        # trying to reduce spikes in sphere mesh since curvature grows quickly
                        # if eps_reduction:
                        #     H_scale = np.mean(np.abs(H_mesh[H_mesh != 0]))
                        #     eps = eps/(1 + H_scale)

                        new_val = f_mesh[i][j] + eps*H_mesh[i][j]*nu_mesh[i][j]
                        f_mesh[i][j] = new_val
                    else:
                        continue


        # seam consistency...
        # f_mesh[N-1] = f_mesh[0]

        counter += 1
        print('TOTAL = ', H_val_total)

        if (np.isnan(H_val_total) | np.isinf(H_val_total)):
            not_minimal = False # break out


    return f_mesh, H_mesh, nu_mesh, plotly_frames, had_singularity, det_gij_min_idx


def run_minimizer_blender(N, u_range, v_range, du, dv, U, V, f_mesh, H_mesh, nu_mesh, tol_eps, eps, frame_freq, fix_u_boundary=True, fix_v_boundary=True, periodic=False):

    not_minimal = True
    counter = 0
    blender_frames = []

    while not_minimal:
        print("At iter = ", counter)

        # add our mesh to blender_frames
        if counter % frame_freq == 0:
            blender_frames.append(f_mesh.copy())

        df_dv = np.gradient(f_mesh, dv, axis=1)
        df_dv2 = np.gradient(df_dv, dv, axis=1)

        if periodic:
            df_du = (np.roll(f_mesh, -1, axis=0) - np.roll(f_mesh, 1, axis=0)) / (2*du)
            df_du2 = (np.roll(f_mesh, -1, axis=0) - 2*f_mesh + np.roll(f_mesh, 1, axis=0)) / (du**2)
            df_dudv = (np.roll(df_dv, -1, axis=0) - np.roll(df_dv, 1, axis=0)) / (2*du)
        else:
            df_du = np.gradient(f_mesh, du, axis=0)
            df_du2 = np.gradient(df_du, du, axis=0)
            df_dudv = np.gradient(df_du, dv, axis=1)

        # run our normal variation iteration
        # compute H at each point on our current surface
        not_minimal_iter = False
        H_val_total = 0
        # NOTE: fixed at the boundary by requirements of minimal surfaces
        # TODO: is this true
        # TODO: neck pinch singularity don't fix?
        Nv = f_mesh.shape[1]
        i_range = range(N) if (periodic or not fix_u_boundary) else range(1, N-1)
        j_range = range(1, Nv-1) if fix_v_boundary else range(Nv)
        for i in i_range:
            for j in j_range:
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

                if np.isnan(H_val):
                    continue # we are at a bad point (not singularity though)

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
                for j in range(f_mesh.shape[1]):
                    H_ij = H_mesh[i][j]
                    if abs(H_ij) > tol_eps:
                        new_val = f_mesh[i][j] + eps*H_mesh[i][j]*nu_mesh[i][j]
                        f_mesh[i][j] = new_val
                    else:
                        continue


        counter += 1
        print('TOTAL = ', H_val_total)

        if (np.isnan(H_val_total) | np.isinf(H_val_total)):
            not_minimal = False # break out


    return f_mesh, H_mesh, nu_mesh, blender_frames

# SEMI-IMPLICIT
def run_implicit(N, u_range, v_range, du, dv, U, V, f_mesh, H_mesh, nu_mesh, tol_eps, eps, frame_freq, fix_u_boundary=True, fix_v_boundary=True, periodic=False):

    not_minimal = True
    counter = 0
    plotly_frames = []

    while not_minimal:

        print("At iter = ", counter)

        # add our mesh to plotly_frames
        if counter % frame_freq == 0:
            f_plot = np.concatenate([f_mesh, f_mesh[0:1]], axis=0) if periodic else f_mesh
            frame = go.Frame(
                    data = go.Surface(
                        x=f_plot[:,:,0],
                        y=f_plot[:,:,1],
                        z=f_plot[:,:,2],
                        colorscale='Viridis',
                        cmin=-1,cmax=1),
                    name = str(counter)
                )
            plotly_frames.append(frame)

        df_dv = np.gradient(f_mesh, dv, axis=1)
        df_dv2 = np.gradient(df_dv, dv, axis=1)

        if periodic:
            df_du = (np.roll(f_mesh, -1, axis=0) - np.roll(f_mesh, 1, axis=0)) / (2*du)
            df_du2 = (np.roll(f_mesh, -1, axis=0) - 2*f_mesh + np.roll(f_mesh, 1, axis=0)) / (du**2)
            df_dudv = (np.roll(df_dv, -1, axis=0) - np.roll(df_dv, 1, axis=0)) / (2*du)
        else:
            df_du = np.gradient(f_mesh, du, axis=0)
            df_du2 = np.gradient(df_du, du, axis=0)
            df_dudv = np.gradient(df_du, dv, axis=1)


        # run our normal variation iteration
        # compute H at each point on our current surface
        not_minimal_iter = False
        H_val_total = 0

        Nv = f_mesh.shape[1]
        i_range = range(N) if (periodic or not fix_u_boundary) else range(1, N-1)
        j_range = range(1, Nv-1) if fix_v_boundary else range(Nv)
        for i in i_range:
            for j in j_range:
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

                if np.isnan(H_val):
                    continue # we are at a bad point (not singularity though)

                H_val_total += H_val**2

                if abs(H_val) > tol_eps:
                    # update my H_mesh because we need to do another iter
                    not_minimal_iter = True # if we are minimal, will never hit this line

                if not_minimal_iter == False:
                    not_minimal = False # break out
                else:
                    not_minimal = True

        # we are done with sweeping through this mesh
        # if eps_reduction:
        #     r = np.mean(np.linalg.norm(f_mesh, axis=2))
        #     if r < 0.9:
        #         not_minimal = False

        # if we are still not minimal, then we need to update the mesh
        if not_minimal:
            for i in range(N):
                for j in range(f_mesh.shape[1]):
                    H_ij = H_mesh[i][j]
                    if abs(H_ij) > tol_eps:
                        # trying to reduce spikes in sphere mesh since curvature grows quickly
                        # if eps_reduction:
                        #     H_scale = np.mean(np.abs(H_mesh[H_mesh != 0]))
                        #     eps = eps/(1 + H_scale)

                        new_val = f_mesh[i][j] + eps*H_mesh[i][j]*nu_mesh[i][j]
                        f_mesh[i][j] = new_val
                    else:
                        continue


        # seam consistency...
        # f_mesh[N-1] = f_mesh[0]

        counter += 1
        print('TOTAL = ', H_val_total)

        if (np.isnan(H_val_total) | np.isinf(H_val_total)):
            not_minimal = False # break out


    return f_mesh, H_mesh, nu_mesh, plotly_frames


def generate_animation(f_plot, plotly_frames, dur):
    fig = go.Figure(
            data = go.Surface(
                x=f_plot[:,:,0],
                y=f_plot[:,:,1],
                z=f_plot[:,:,2],
                colorscale='Viridis',
                cmin=-1,cmax=1
            ),
            frames = plotly_frames
        )

    fig.update_layout(
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=0,
                    x=0.5,
                    xanchor='center',
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(frame=dict(duration=dur, redraw=True), fromcurrent=True)]
                            )
                        ]
                    )
                ]
            )
    
    fig.update_layout(
            sliders=[
                dict(
                    steps=[
                        dict(
                            method='animate',
                            args=[[frame.name], dict(mode='immediate', frame=dict(duration=dur, redraw=True))],
                            label=str(i*10)
                            )
                        for i, frame in enumerate(plotly_frames)
                        ],
                    currentvalue=dict(prefix='Iteration: '),
                    x=0.1,
                    len=0.9
                    )
                ]
            )

    fig.show()
