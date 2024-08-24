import argparse
import os.path

import cupy
import numpy as np
from fastqslpy import FastQSL
from fastqslpy.kernels import compileTraceBlineAdaptive
from tqdm import tqdm

from nf2.evaluation.output import CartesianOutput
from nf2.evaluation.vtk import save_vtk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--nf2_path', type=str, help='path to the source NF2 file')
    parser.add_argument('--result_path', type=str, help='path to the target VTK file', required=False, default=None)
    parser.add_argument('--height_range', type=float, nargs=2, help='height range in Mm', required=False, default=None)

    args = parser.parse_args()
    nf2_path = args.nf2_path

    result_path = args.result_path
    height_range = args.height_range
    os.makedirs(result_path, exist_ok=True)

    nf2_out = CartesianOutput(nf2_path)
    output = nf2_out.load_cube(Mm_per_pixel=0.72, height_range=height_range, progress=True)

    # convert B to cuda array
    Bx = output["b"][..., 0].astype(np.float32)
    By = output["b"][..., 1].astype(np.float32)
    Bz = output["b"][..., 2].astype(np.float32)

    # convert to fortran order
    Bx_gpu = cupy.array(Bx)
    By_gpu = cupy.array(By)
    Bz_gpu = cupy.array(Bz)

    # convert to cuda array
    Bx_gpu = cupy.asfortranarray(Bx_gpu)
    By_gpu = cupy.asfortranarray(By_gpu)
    Bz_gpu = cupy.asfortranarray(Bz_gpu)

    # convert J to cuda array
    Jx = output["j"][..., 0].astype(np.float32)
    Jy = output["j"][..., 1].astype(np.float32)
    Jz = output["j"][..., 2].astype(np.float32)

    # convert to fortran order
    Jx_gpu = cupy.array(Jx)
    Jy_gpu = cupy.array(Jy)
    Jz_gpu = cupy.array(Jz)

    # convert to cuda array
    Jx_gpu = cupy.asfortranarray(Jx_gpu)
    Jy_gpu = cupy.asfortranarray(Jy_gpu)
    Jz_gpu = cupy.asfortranarray(Jz_gpu)

    trace_all_b_line = compileTraceBlineAdaptive()

    # prepare variables
    BshapeN = np.zeros(3, dtype=np.int32)
    BshapeN[:] = Bx.shape
    print(BshapeN)
    BshapeN = cupy.array(BshapeN)

    interp_ratio = 3
    stride_step = 1 / interp_ratio
    x_range = [0, Bx.shape[0]]
    y_range = [0, Bx.shape[1]]
    z_range = [0, Bx.shape[2]]
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
                          Jx_gpu, Jy_gpu, Jz_gpu, twist_all, flag_twist,
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

        (x_start, y_start, z_start, x_end, y_end, z_end, z_inp,
         Bx_start, By_start, Bz_start, Bx_end, By_end, Bz_end, Bx_inp, By_inp, Bz_inp
         ) = [None for _ in range(16)]
        (B_flag, flag_start, flag_end) = [None for _ in range(3)]
        pinned_mempool.free_all_blocks()

    Qube = Qube[::interp_ratio, ::interp_ratio, ::interp_ratio]
    Twube = Twube[::interp_ratio, ::interp_ratio, ::interp_ratio]

    base_name = os.path.basename(nf2_path).split('.')[0]
    save_vtk(os.path.join(result_path, f"{base_name}.vtk"),
             scalars={"Q": Qube, "Twist": Twube},
             vectors={"B": output["b"]})
