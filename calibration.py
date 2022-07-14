import cv2
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as onp
from jax import vmap
from jaxlie import SE3

print("Calculate calibration data...")
# Calibration for rectifying the pictures
plt.style.use('dark_background')

pts_3d = onp.loadtxt('/content/drive/MyDrive/Calibration/calibration_info/3d_pts.txt')
pts_2d_a = onp.loadtxt('/content/drive/MyDrive/Calibration/calibration_info/2d_pts_a.txt')
pts_2d_b = onp.loadtxt('/content/drive/MyDrive/Calibration/calibration_info/2d_pts_b.txt')

f = 5.084752337001038995e+02  # 1060.31
c_x = 3.225185456035997049e+02
c_y = 2.505901000840876520e+02


def main():
    def reprojection_error_bearing(T_b_a, u_a, v_a, u_b, v_b):
        bearings_a = jnp.array([u_a - c_x, v_a - c_y, f])
        bearings_b = jnp.array([u_b - c_x, v_b - c_y, f])

        R = T_b_a.rotation()

        err = jnp.mean((R @ bearings_a - bearings_b) ** 2)

        return err

    def reprojection_error_point(T_c_w, p_w, u, v):
        p_c = T_c_w.apply(p_w)

        x = p_c[0]
        y = p_c[1]
        z = p_c[2]

        u_p = f * x / z + c_x
        v_p = f * y / z + c_y

        err = jnp.mean((u - u_p) ** 2 + (v - v_p) ** 2)

        return err

    def compute_total_error(params, u_a, v_a, u_b, v_b, u_a_stars, v_a_stars, u_b_stars, v_b_stars, p_w):
        T_a_w = SE3.exp(params[:6])
        T_b_w = SE3.exp(params[6:])

        err_1 = m_reprojection_error_point(T_a_w, p_w, u_a, v_a)
        err_2 = m_reprojection_error_point(T_b_w, p_w, u_b, v_b)
        err_3 = m_reprojection_error_bearing(T_b_w @ T_a_w.inverse(), u_a_stars, v_a_stars, u_b_stars, v_b_stars)

        err_4 = (T_a_w.inverse().translation() - jnp.array([1.259, -1.14, 0.808])) ** 2

        total_err = 0.5 * jnp.mean(err_1) + 0.5 * jnp.mean(err_2) + 3.0 * jnp.mean(err_3) + jnp.sum(err_4)

        return total_err / 1e5

    def plot_points(T_c_w, p_w):
        p_c = vmap(T_c_w.apply)(p_w)

        x = p_c[:, 0]
        y = p_c[:, 1]
        z = p_c[:, 2]

        u_p = f * x / z + c_x
        v_p = f * y / z + c_y

        im1 = cv2.imread("/content/drive/MyDrive/Calibration/calibration_info/tl4_2021-10-30_230001_CAL1.jpg")

        plt.scatter(onp.asarray(u_p), onp.asarray(v_p), c='r')
        plt.imshow(im1)
        plt.show()

    twist = onp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    T_a_w = SE3.exp(twist)

    twist = onp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    T_b_w = SE3.exp(twist)

    u_a = jnp.asarray(pts_2d_a[16:, 0])
    v_a = jnp.asarray(pts_2d_a[16:, 1])

    u_b = jnp.asarray(pts_2d_b[16:, 0])
    v_b = jnp.asarray(pts_2d_b[16:, 1])

    u_a_stars = jnp.asarray(pts_2d_a[:16, 0])
    v_a_stars = jnp.asarray(pts_2d_a[:16, 1])

    u_b_stars = jnp.asarray(pts_2d_b[:16, 0])
    v_b_stars = jnp.asarray(pts_2d_b[:16, 1])

    m_reprojection_error_point = vmap(reprojection_error_point, (None, 0, 0, 0), 0)
    m_reprojection_error_bearing = vmap(reprojection_error_bearing, (None, 0, 0, 0, 0), 0)

    p_w = jnp.asarray(pts_3d)

    params = onp.array(
        [1.3690165, 1.4032199, -0.0737519, 1.1455071, 0.09243497, -0.11357018, 0.20850165, 1.001657, 0.03703392,
         1.1190069,
         0.08403286, -0.12874132])

    # err = compute_total_error(params, u_a, v_a, u_b, v_b, u_a_stars, v_a_stars, u_b_stars, v_b_stars, p_w )

    # error_grad = grad(compute_total_error)
    # delta_param = error_grad(params, u_a, v_a, u_b, v_b, u_a_stars, v_a_stars, u_b_stars, v_b_stars, p_w)

    maxiter = 0
    solver = jaxopt.LBFGS(fun=compute_total_error, maxiter=maxiter, tol=0.0001, verbose=True)
    res = solver.run(params, u_a=u_a, v_a=v_a, u_b=u_b, v_b=v_b, u_a_stars=u_a_stars, v_a_stars=v_a_stars,
                     u_b_stars=u_b_stars, v_b_stars=v_b_stars, p_w=p_w)

    params, state = res
    print(params)

    T_a_w = SE3.exp(params[:6])
    T_b_w = SE3.exp(params[6:])

    K = jnp.array([[f, 0., c_x], [0., f, c_y], [0., 0., 1.]])

    T_b_a = T_b_w @ T_a_w.inverse()
    R = T_b_a.rotation()
    t = T_b_a.translation()

    print('Cam 1 position: ', T_a_w.inverse().translation())
    print('Cam 1 orientation: ', T_a_w.inverse().rotation().as_matrix())

    print('Cam 2 position: ', T_b_w.inverse().translation())
    print('Cam 2 orientation: ', T_b_w.inverse().rotation().as_matrix())

    tx = jnp.array([[0., -t[2], t[1]], [t[2], 0., -t[0]], [-t[1], t[0], 0.]])

    F = jnp.linalg.inv(K.T) @ R.as_matrix() @ tx @ jnp.linalg.inv(K)

    print(F)

    onp.savetxt('F.txt', F)
    # plot_points(T_a_w, p_w)

    im1 = cv2.resize(cv2.imread("/content/drive/MyDrive/Calibration/calibration_info/tl4_2021-10-30_230001_CAL1.jpg"),
                     (640, 480))

    camera_matrix_l = onp.asarray(K).astype(onp.float64)
    camera_matrix_r = onp.asarray(K).astype(onp.float64)
    dist_coeffs_l = onp.zeros(5).astype(onp.float64)
    dist_coeffs_r = onp.zeros(5).astype(onp.float64)
    R_ab = onp.asarray(R.as_matrix()).astype(onp.float64)
    T = onp.asarray(t).astype(onp.float64)
    T = T / onp.linalg.norm(T) * 61.0

    print('R_ab', R_ab)
    print('T', T)

    RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(
        camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r,
        im1[:, :, 0].shape[::-1], R_ab, T, alpha=-1)

    mapL1, mapL2 = cv2.initUndistortRectifyMap(
        camera_matrix_l, dist_coeffs_l, RL, PL, im1[:, :, 0].shape[::-1],
        cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(
        camera_matrix_r, dist_coeffs_r, RR, PR, im1[:, :, 0].shape[::-1],
        cv2.CV_32FC1)

    return mapL1, mapL2, mapR1, mapR2


if __name__ == "__main__":
    main()
