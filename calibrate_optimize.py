import cv2
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as onp
from jax import vmap
from jaxlie import SE3, SO3
from matplotlib import colors

plt.style.use('dark_background')

pts_3d = onp.loadtxt('3d_pts.txt')
pts_2d_a = onp.loadtxt('2d_pts_a.txt')
pts_2d_b = onp.loadtxt('2d_pts_b.txt')

f = 5.084752337001038995e+02  # 1060.31
c_x = 3.225185456035997049e+02
c_y = 2.505901000840876520e+02


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
    [1.3690165, 1.4032199, -0.0737519, 1.1455071, 0.09243497, -0.11357018, 0.20850165, 1.001657, 0.03703392, 1.1190069,
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

cap1 = cv2.VideoCapture('videos/tl4_2021-10-24_12A.mp4')
cap2 = cv2.VideoCapture('videos/tl_2021-10-24_12A.mp4')

im1 = cv2.resize(cv2.imread("tl4_2021-10-30_230001_CAL1.jpg"), (640, 480))

frame_no = 0

# for i in range(70):
#     ret, im1 = cap1.read()
#     ret, im2 = cap2.read()

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

# calib_im1 = cv2.imread('tl4_2021-10-30_230001_CAL1.jpg')
# calib_im2 = cv2.imread('tl_2021-10-30_230002_CAL1.jpg')
# calib_undistorted_rectifiedL = cv2.remap(calib_im1, mapL1, mapL2, cv2.INTER_LINEAR)
# calib_undistorted_rectifiedR = cv2.remap(calib_im2, mapR1, mapR2, cv2.INTER_LINEAR)
# cv2.imwrite("rectified_1.png", calib_undistorted_rectifiedL)
# cv2.imwrite("rectified_2.png", calib_undistorted_rectifiedR)

# # for i in range(220):
# #     ret, im1 = cap1.read()
# #     ret, im2 = cap2.read()
# # frame_no = 220

# maximumDisparities = 64
# left_matcher = cv2.StereoSGBM_create(0, maximumDisparities, blockSize=9, uniquenessRatio=1, speckleWindowSize=11,
#                                     speckleRange=10, disp12MaxDiff=40)
# # right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# lmbda = 5
# sigma = 0.01
# wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
# wls_filter.setLambda(lmbda)
# wls_filter.setSigmaColor(sigma)

# while cap1.isOpened():
#     ret, im1 = cap1.read()
#     ret, im2 = cap2.read()

#     # im1 = cv2.imread('/home/ronnie/Projects/StereoClouds/sim_cloud/cam1.png')
#     # im2 = cv2.imread('/home/ronnie/Projects/StereoClouds/sim_cloud/cam2.png')
#     frame_no += 1
#     im1 = cv2.resize(im1, (640, 480))  # cv2.resize(cv2.imread("tl4_2021-10-24_12A.jpg"), (1280,960))
#     im2 = cv2.resize(im2, (640, 480))  # cv2.imread("tl_2021-10-24_12A.jpg")

#     # pts1 = onp.stack([u_a_stars, v_a_stars],1)
#     # pts2 = onp.stack([u_b_stars, v_b_stars],1)

#     # h1, w1 = im1.shape[:2]
#     # h2, w2 = im2.shape[:2]
#     # _, H1, H2 = cv2.stereoRectifyUncalibrated(
#     #     onp.float32(pts1), onp.float32(pts2), onp.asarray(F), imgSize=(w1, h1)
#     # )
#     # img1_rectified = cv2.warpPerspective(im1, H1, (w1, h1))
#     # img2_rectified = cv2.warpPerspective(im2, H2, (w2, h2))

#     undistorted_rectifiedL = cv2.remap(im1, mapL1, mapL2, cv2.INTER_LINEAR)
#     undistorted_rectifiedR = cv2.remap(im2, mapR1, mapR2, cv2.INTER_LINEAR)

#     disparity = left_matcher.compute(undistorted_rectifiedL[:, :, 2].astype(onp.uint8),
#                                      undistorted_rectifiedR[:, :, 2].astype(onp.uint8))
#     cv2.filterSpeckles(disparity, 0, 256, maximumDisparities)

#     # disparity = disparity.astype(onp.float32) / 16.0

#     rimg1 = cv2.cvtColor(undistorted_rectifiedL, cv2.COLOR_BGR2GRAY).astype(onp.uint8)
#     rimg2 = cv2.cvtColor(undistorted_rectifiedR, cv2.COLOR_BGR2GRAY).astype(onp.uint8)

#     # displ = left_matcher.compute(rimg1, rimg2)
#     # dispr = right_matcher.compute(rimg2, rimg1)
#     # displ = onp.int16(displ)
#     # dispr = onp.int16(dispr)
#     # disparity = dispr #wls_filter.filter(displ, rimg1, None, dispr)
#     disparity = disparity.astype(onp.float32) / 16.0

#     mask = (disparity > 0).astype(onp.float64) * (disparity < 2000).astype(onp.float64)

#     p_c = cv2.reprojectImageTo3D(disparity, Q)
#     p_c = jnp.asarray(p_c.reshape(-1, 3))

#     RL_ = SO3.from_matrix(RL)

#     p_c = vmap(RL_.inverse().apply)(p_c)
#     p_w = vmap(T_a_w.inverse().apply)(p_c)

#     rgb_flat = undistorted_rectifiedL[:, :, ::-1].reshape(-1, 3)
#     # onp.savetxt('out/points_{:04d}.txt'.format(frame_no), onp.concatenate([ onp.nan_to_num(onp.asarray(p_w)), onp.asarray(rgb_flat)],1) )
#     p_w = onp.asarray(p_w.reshape(480, 640, 3))

#     # plt.subplot(1,2,1)
#     # plt.imshow(undistorted_rectifiedL[:,:,::-1])
#     # plt.subplot(1,2,2)
#     # plt.imshow(undistorted_rectifiedR[:,:,::-1])

#     cv2.imwrite('/home/ronnie/Pictures/clouds/rectified/left/img_{:04d}.png'.format(frame_no), undistorted_rectifiedL)
#     cv2.imwrite('/home/ronnie/Pictures/clouds/rectified/right/img_{:04d}.png'.format(frame_no), undistorted_rectifiedR)
#     # onp.save('/home/ronnie/Projects/StereoClouds/sim_cloud/sgm_result.npy',disparity)

#     plt.imshow(undistorted_rectifiedL[:, :, ::-1])
#     plt.imshow(p_w[:, :, 2], vmin=100, vmax=20000, norm=colors.LogNorm(), cmap='plasma')
#     plt.axis('off')
#     plt.colorbar()

#     # plt.show()
#     plt.savefig('out/im_{:04d}.png'.format(frame_no))
#     plt.clf()

#     print("Finished!")

#     # cv2.imwrite("rectified_1.png", undistorted_rectifiedL)
#     # cv2.imwrite("rectified_2.png", undistorted_rectifiedR)
