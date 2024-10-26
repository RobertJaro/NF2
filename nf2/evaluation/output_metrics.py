import numpy as np
from astropy import constants, units as u
from astropy.nddata import block_reduce
from tqdm import tqdm

from nf2.data.util import cartesian_to_spherical
from nf2.evaluation.energy import get_free_mag_energy
from nf2.train.model import calculate_current_from_jacobian


def current_density(jac_matrix, **kwargs):
    j = calculate_current_from_jacobian(jac_matrix, f=np) * constants.c / (4 * np.pi)
    return {'j': j.to(u.G / u.s)}


def b_nabla_bz(b, jac_matrix, **kwargs):
    # compute B * nabla * Bz
    bx = b[..., 0]
    by = b[..., 1]
    bz = b[..., 2]
    dBx_dx = jac_matrix[..., 0, 0]
    dBx_dy = jac_matrix[..., 0, 1]
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dx = jac_matrix[..., 1, 0]
    dBy_dy = jac_matrix[..., 1, 1]
    dBy_dz = jac_matrix[..., 1, 2]
    dBz_dx = jac_matrix[..., 2, 0]
    dBz_dy = jac_matrix[..., 2, 1]
    dBz_dz = jac_matrix[..., 2, 2]
    #
    norm_B = np.linalg.norm(b, axis=-1) + 1e-10 * b.unit

    dnormB_dx = - norm_B ** -3 * (bx * dBx_dx + by * dBy_dx + bz * dBz_dx)
    dnormB_dy = - norm_B ** -3 * (bx * dBx_dy + by * dBy_dy + bz * dBz_dy)
    dnormB_dz = - norm_B ** -3 * (bx * dBx_dz + by * dBy_dz + bz * dBz_dz)

    b_nabla_bz = (bx * dBz_dx + by * dBz_dy + bz * dBz_dz) / norm_B ** 2 + \
                 (bz / norm_B) * (bx * dnormB_dx + by * dnormB_dy + bz * dnormB_dz)
    b_nabla_bz = b_nabla_bz.to_value(1 / u.Mm)
    return {'b_nabla_bz': b_nabla_bz}


def magnetic_helicity(b, a, **kwargs):
    helicity = np.sum(a * b, axis=-1)
    return {'magnetic_helicity': helicity}


def alpha(b, jac_matrix, **kwargs):
    j = calculate_current_from_jacobian(jac_matrix, f=np)
    alpha = np.linalg.norm(j, axis=-1) / np.linalg.norm(b, axis=-1)
    # threshold = np.linalg.norm(b, axis=-1).to_value(u.G) < 1  # set alpha to 0 for weak fields (coronal holes)
    # alpha[threshold] = 0 * u.Mm ** -1
    alpha = alpha.to(u.Mm ** -1)
    return {'alpha': alpha}


def spherical_energy_gradient(b, jac_matrix, coords, **kwargs):
    dBx_dx = jac_matrix[..., 0, 0]
    dBy_dx = jac_matrix[..., 1, 0]
    dBz_dx = jac_matrix[..., 2, 0]
    dBx_dy = jac_matrix[..., 0, 1]
    dBy_dy = jac_matrix[..., 1, 1]
    dBz_dy = jac_matrix[..., 2, 1]
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dz = jac_matrix[..., 1, 2]
    dBz_dz = jac_matrix[..., 2, 2]
    # E = b^2 = b_x^2 + b_y^2 + b_z^2
    # dE/dx = 2 * (b_x * dBx_dx + b_y * dBy_dx + b_z * dBz_dx)
    # dE/dy = 2 * (b_x * dBx_dy + b_y * dBy_dy + b_z * dBz_dy)
    # dE/dz = 2 * (b_x * dBx_dz + b_y * dBy_dz + b_z * dBz_dz)
    dE_dx = 2 * (b[..., 0] * dBx_dx + b[..., 1] * dBy_dx + b[..., 2] * dBz_dx)
    dE_dy = 2 * (b[..., 0] * dBx_dy + b[..., 1] * dBy_dy + b[..., 2] * dBz_dy)
    dE_dz = 2 * (b[..., 0] * dBx_dz + b[..., 1] * dBy_dz + b[..., 2] * dBz_dz)

    coords_spherical = cartesian_to_spherical(coords, f=np)
    t = coords_spherical[..., 1]
    p = coords_spherical[..., 2]
    dE_dr = (np.sin(t) * np.cos(p)) * dE_dx + \
            (np.sin(t) * np.sin(p)) * dE_dy + \
            np.cos(p) * dE_dz

    return {'dE_dr': dE_dr}


def energy_gradient(b, jac_matrix, **kwargs):
    dBx_dz = jac_matrix[..., 0, 2]
    dBy_dz = jac_matrix[..., 1, 2]
    dBz_dz = jac_matrix[..., 2, 2]
    # E = b^2 = b_x^2 + b_y^2 + b_z^2
    # dE/dz = 2 * (b_x * dBx_dz + b_y * dBy_dz + b_z * dBz_dz)
    dE_dz = 2 * (b[..., 0] * dBx_dz + b[..., 1] * dBy_dz + b[..., 2] * dBz_dz)

    return {'dE_dz': dE_dz}


def los_trv_azi(b, **kwargs):
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    b_los = bz.to_value(u.G)
    b_trv = np.sqrt(bx ** 2 + by ** 2).to_value(u.G)
    azimuth = np.arctan2(by, bx).to_value(u.deg)
    b_los_trv_azi = np.stack([b_los, b_trv, azimuth], -1)
    return {'b_los_trv_azi': b_los_trv_azi}


def free_energy(b, **kwargs):
    free_energy = get_free_mag_energy(b.to_value(u.G)) * u.erg * u.cm ** -3
    return {'free_energy': free_energy}


def squashing_factor(b, interp_ratio = 3, x_range=None, y_range=None, z_range=None, **kwargs):
    # local imports for optional dependency
    import cupy
    from fastqslpy import FastQSL
    from fastqslpy.kernels import compileTraceBlineAdaptive

    # convert B to cuda array
    Bx = b[..., 0].astype(np.float32)
    By = b[..., 1].astype(np.float32)
    Bz = b[..., 2].astype(np.float32)
    # convert to fortran order
    Bx_gpu = cupy.array(Bx)
    By_gpu = cupy.array(By)
    Bz_gpu = cupy.array(Bz)
    # convert to cuda array
    Bx_gpu = cupy.asfortranarray(Bx_gpu)
    By_gpu = cupy.asfortranarray(By_gpu)
    Bz_gpu = cupy.asfortranarray(Bz_gpu)
    # calc curl(B)
    curBx_gpu = cupy.zeros_like(Bx_gpu)
    curBy_gpu = cupy.zeros_like(By_gpu)
    curBz_gpu = cupy.zeros_like(Bz_gpu)

    curBx_gpu[:, 1:-1, 1:-1] = ((Bz_gpu[:, 2:, 1:-1] - Bz_gpu[:, 0:-2, 1:-1]) / 2.
                                - (By_gpu[:, 1:-1, 2:] - By_gpu[:, 1:-1, 0:-2]) / 2)
    curBy_gpu[1:-1, :, 1:-1] = ((Bx_gpu[1:-1, :, 2:] - Bx_gpu[1:-1, :, 0:-2]) / 2.
                                - (Bz_gpu[2:, :, 1:-1] - Bz_gpu[0:-2, :, 1:-1]) / 2)
    curBz_gpu[1:-1, 1:-1, :] = ((By_gpu[2:, 1:-1, :] - By_gpu[0:-2, 1:-1, :]) / 2.
                                - (Bx_gpu[1:-1, 2:, :] - Bx_gpu[1:-1, 0:-2, :]) / 2)

    # take care of z=0
    curBx_gpu[1:-1, 1:-1, 0] = ((Bz_gpu[1:-1, 2:, 0] - Bz_gpu[1:-1, 0:-2, 0]) / 2.
                                - (-3. * By_gpu[1:-1, 1:-1, 0] + 4. * By_gpu[1:-1, 1:-1, 1] - By_gpu[1:-1, 1:-1,
                                                                                              2]) / 2)
    curBy_gpu[1:-1, 1:-1, 0] = ((-3. * Bx_gpu[1:-1, 1:-1, 0] + 4. * Bx_gpu[1:-1, 1:-1, 1] - Bx_gpu[1:-1, 1:-1, 2]) / 2.
                                - (Bz_gpu[2:, 1:-1, 0] - Bz_gpu[0:-2, 1:-1, 0]) / 2)

    trace_all_b_line = compileTraceBlineAdaptive()
    # prepare variables
    BshapeN = np.zeros(3, dtype=np.int32)
    BshapeN[:] = Bx.shape
    BshapeN = cupy.array(BshapeN)
    stride_step = 1 / interp_ratio
    x_range = [0, Bx.shape[0]] if x_range is None else x_range
    y_range = [0, Bx.shape[1]] if y_range is None else y_range
    z_range = [0, Bx.shape[2]] if z_range is None else z_range
    # z_range = [0,1]
    x_i = cupy.linspace(*x_range, np.uint(interp_ratio * (x_range[1] - x_range[0])), dtype=cupy.float32)
    y_i = cupy.linspace(*y_range, np.uint(interp_ratio * (y_range[1] - y_range[0])), dtype=cupy.float32)
    z_i = cupy.linspace(*z_range, np.uint(interp_ratio * (z_range[1] - z_range[0])), dtype=cupy.float32)
    flag_twist = cupy.array([True], dtype=cupy.bool_)
    flag_twist_false = cupy.array([False], dtype=cupy.bool_)
    dummy = cupy.zeros([1, 1], dtype=cupy.float32)
    dummy64 = cupy.zeros([1, 1], dtype=cupy.float64)
    x_arr, y_arr = cupy.meshgrid(x_i, y_i)
    xy_shape = x_arr.shape
    x_inp = x_arr.flatten()
    y_inp = y_arr.flatten()

    line_len = cupy.zeros(x_inp.shape, cupy.float64)
    N = cupy.array([x_inp.shape[0]], cupy.ulonglong)
    s_len = cupy.array([1. / 16.], cupy.float32)
    tol_coef = cupy.array([cupy.sqrt(0.1)], cupy.float32)
    inp_norm = cupy.array([0, 0, 1.], cupy.float32)
    twist_all = cupy.zeros(x_inp.shape, cupy.float64)
    blck = (128, 1, 1)
    grd = (28, 1)
    cupy.cuda.stream.get_current_stream().synchronize()
    Qube = np.zeros([xy_shape[1], xy_shape[0], z_i.shape[0]], dtype=np.float32)
    Twube = np.zeros([xy_shape[1], xy_shape[0], z_i.shape[0]], dtype=np.float32)
    Liube = np.zeros([xy_shape[1], xy_shape[0], z_i.shape[0]], dtype=np.float32)
    pinned_mempool = cupy.get_default_pinned_memory_pool()
    for idx_pos_z, z_pos in tqdm(enumerate(z_i), total=z_i.shape[0], desc='Compute Q and T'):

        x_inp = x_arr.flatten()
        y_inp = y_arr.flatten()
        z_inp = (cupy.zeros_like(x_inp, cupy.float32) + z_pos)

        (x_start, y_start, z_start, x_end, y_end, z_end,
         Bx_start, By_start, Bz_start, Bx_end, By_end, Bz_end, Bx_inp, By_inp, Bz_inp
         ) = [cupy.zeros(x_inp.shape, dtype=cupy.float32) for _ in range(15)]

        (B_flag, flag_start, flag_end) = [cupy.zeros(x_inp.shape, dtype=cupy.int32) for _ in range(3)]

        cupy.cuda.stream.get_current_stream().synchronize()
        # run the big calculation
        trace_all_b_line(blck, grd,
                         (Bx_gpu, By_gpu, Bz_gpu, BshapeN,
                          curBx_gpu, curBy_gpu, curBz_gpu, twist_all, flag_twist,
                          x_inp, y_inp, z_inp, inp_norm,
                          x_start, y_start, z_start, flag_start,
                          x_end, y_end, z_end, flag_end,
                          Bx_inp, By_inp, Bz_inp, B_flag,
                          Bx_start, By_start, Bz_start,
                          Bx_end, By_end, Bz_end,
                          s_len, N, line_len, tol_coef))
        cupy.cuda.stream.get_current_stream().synchronize()
        (x_end_arr, y_end_arr, z_end_arr, flag_end_arr,
         x_start_arr, y_start_arr, z_start_arr, flag_start_arr,
         Bx_in_arr, By_in_arr, Bz_in_arr,
         Bx_out_arr, By_out_arr, Bz_out_arr,
         Bx_0_arr, By_0_arr, Bz_0_arr, B_flag_arr, twist_all_arr,
         line_len_arr) = FastQSL.ResReshape(xy_shape,
                                            x_end, y_end, z_end, flag_end,
                                            x_start, y_start, z_start, flag_start,
                                            Bx_start, By_start, Bz_start,
                                            Bx_end, By_end, Bz_end,
                                            Bx_inp, By_inp, Bz_inp, B_flag, twist_all, line_len)

        if z_pos < 1e-5:
            B0cube = Bz_0_arr.get()

        cupy.cuda.stream.get_current_stream().synchronize()
        Q = FastQSL.QCalcPlane(x_end_arr, y_end_arr, z_end_arr, flag_end_arr,
                               x_start_arr, y_start_arr, z_start_arr, flag_start_arr,
                               Bx_in_arr, By_in_arr, Bz_in_arr,
                               Bx_out_arr, By_out_arr, Bz_out_arr,
                               Bx_0_arr, By_0_arr, Bz_0_arr,
                               B_flag_arr, stride_step)
        stride_this = cupy.float32(1. / interp_ratio) / 8

        (cut_inp_x, cut_inp_y, cut_inp_z,
         cut_start_x, cut_start_y, cut_start_z, flag_cut_start,
         cut_end_x, cut_end_y, cut_end_z, flag_cut_end,
         Bx_inp_cut, By_inp_cut, Bz_inp_cut, B_flag_cut,
         Bx_start_cut, By_start_cut, Bz_start_cut,
         Bx_end_cut, By_end_cut, Bz_end_cut,
         N_cut, line_len_cut, Bz0_start, Bz0_end) = FastQSL.CookPseudoLine(
            x_end_arr, y_end_arr, z_end_arr, flag_end_arr,
            x_start_arr, y_start_arr, z_start_arr, flag_start_arr,
            Bx_in_arr, By_in_arr, Bz_in_arr,
            Bx_out_arr, By_out_arr, Bz_out_arr,
            Bz_0_arr, B_flag_arr, stride_this)

        cupy.cuda.stream.get_current_stream().synchronize()
        trace_all_b_line(blck, grd,
                         (Bx_gpu, By_gpu, Bz_gpu, BshapeN,
                          dummy, dummy, dummy, dummy64, flag_twist_false,
                          cut_inp_x, cut_inp_y, cut_inp_z, inp_norm,
                          cut_start_x, cut_start_y, cut_start_z, flag_cut_start,
                          cut_end_x, cut_end_y, cut_end_z, flag_cut_end,
                          Bx_inp_cut, By_inp_cut, Bz_inp_cut, B_flag_cut,
                          Bx_start_cut, By_start_cut, Bz_start_cut,
                          Bx_end_cut, By_end_cut, Bz_end_cut,
                          s_len, N_cut, line_len_cut, tol_coef * .1))

        cupy.cuda.stream.get_current_stream().synchronize()
        (X1, Y1, X2, Y2) = [cupy.zeros(cut_inp_x.shape, dtype=cupy.float32)
                            for _ in range(4)]

        idx_Z1_cut = (flag_cut_start - 1) // 2 == 2
        idx_Z2_cut = (flag_cut_end - 1) // 2 == 2
        idx_Y1_cut = (flag_cut_start - 1) // 2 == 1
        idx_Y2_cut = (flag_cut_end - 1) // 2 == 1
        idx_X1_cut = (flag_cut_start - 1) // 2 == 0
        idx_X2_cut = (flag_cut_end - 1) // 2 == 0

        cupy.cuda.stream.get_current_stream().synchronize()
        # Z plane
        X1[idx_Z1_cut] = cut_start_x[idx_Z1_cut]
        Y1[idx_Z1_cut] = cut_start_y[idx_Z1_cut]
        X2[idx_Z2_cut] = cut_end_x[idx_Z2_cut]
        Y2[idx_Z2_cut] = cut_end_y[idx_Z2_cut]
        # Y plane
        X1[idx_Y1_cut] = cut_start_z[idx_Y1_cut]
        Y1[idx_Y1_cut] = cut_start_x[idx_Y1_cut]
        X2[idx_Y2_cut] = cut_end_z[idx_Y2_cut]
        Y2[idx_Y2_cut] = cut_end_x[idx_Y2_cut]
        # X plane
        X1[idx_X1_cut] = cut_start_y[idx_X1_cut]
        Y1[idx_X1_cut] = cut_start_z[idx_X1_cut]
        X2[idx_X2_cut] = cut_end_y[idx_X2_cut]
        Y2[idx_X2_cut] = cut_end_z[idx_X2_cut]

        dx2xc = X2[0::4] - X2[2::4]
        dx2yc = X2[1::4] - X2[3::4]
        dy2xc = Y2[0::4] - Y2[2::4]
        dy2yc = Y2[1::4] - Y2[3::4]
        dx1xc = X1[0::4] - X1[2::4]
        dx1yc = X1[1::4] - X1[3::4]
        dy1xc = Y1[0::4] - Y1[2::4]
        dy1yc = Y1[1::4] - Y1[3::4]
        a_cut = (dx2xc * dy1yc - dx2yc * dy1xc)
        b_cut = (dx2yc * dx1xc - dx2xc * dx1yc)
        c_cut = (dy2xc * dy1yc - dy2yc * dy1xc)
        d_cut = (dy2yc * dx1xc - dy2xc * dx1yc)

        bnr_cut = cupy.abs(Bz0_end) / (cupy.abs(Bz0_start)) * ((1 / stride_this / 2) ** 4)
        Qcut = (a_cut ** 2 + b_cut ** 2 + c_cut ** 2 + d_cut ** 2) * bnr_cut
        Qcut[cupy.where(Qcut < 1.0)] = 1.0

        Q_all = cupy.zeros(x_end_arr.shape, dtype=cupy.float32)
        Q_all[1:-1, 1:-1] = Q
        Q_all[B_flag_arr == 1] = Qcut

        Qube[:, :, idx_pos_z] = Q_all.T.get()
        Twube[:, :, idx_pos_z] = twist_all_arr.T.get()
        Liube[:, :, idx_pos_z] = line_len_arr.T.get()

        cupy.cuda.stream.get_current_stream().synchronize()

        pinned_mempool.free_all_blocks()
    Qube = block_reduce(Qube, (interp_ratio, interp_ratio, interp_ratio), func=np.mean)
    Twube = block_reduce(Twube, (interp_ratio, interp_ratio, interp_ratio), func=np.mean)

    return {"q": Qube, "twist": Twube}


metric_mapping = {
    'j': current_density,
    'b_nabla_bz': b_nabla_bz,
    'alpha': alpha,
    'spherical_energy_gradient': spherical_energy_gradient,
    'energy_gradient': energy_gradient,
    'magnetic_helicity': magnetic_helicity,
    'los_trv_azi': los_trv_azi,
    'free_energy': free_energy,
    'squashing_factor': squashing_factor
}
