import argparse
import glob
import os
import re

import cv2
import numpy as np
from matplotlib import pyplot as plt

from funcs import create_dir

id = "0034"


def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, _ = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_BGRA2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_BGRA2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color


def draw_horizontal_lines(img1_rectified, img2_rectified):
    global axes
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(img1_rectified)
    axes[1].imshow(img2_rectified)
    axes[0].axhline(240)
    axes[1].axhline(240)
    axes[0].axhline(450)
    axes[1].axhline(450)
    plt.suptitle("Rectified images")
    plt.savefig("rectified_images.png")
    plt.show()


def main():
    img1 = cv2.imread(r"rendered_train/Left/rgb_{}.PNG".format(id))
    img2 = cv2.imread(r"rendered_train/Right/rgb_{}.PNG".format(id))

    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # imgSift = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("SIFT Keypoints", imgSift)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Keep good matches: calculate distinctive image features Lowe, D.G. Distinctive Image Features from
    # Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004).
    # https://doi.org/10.1023/B:VISI.0000029664.99615.94 https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask[300:500],
    #                    flags=cv2.DrawMatchesFlags_DEFAULT)
    #
    # keypoint_matches = cv2.drawMatchesKnn(
    #     img1, kp1, img2, kp2, matches[300:500], None, **draw_params)
    # cv2.imshow("Keypoint matches", keypoint_matches)
    # cv2.waitKey(0)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    # lines1 = cv2.computeCorrespondEpilines(
    #     pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
    # lines1 = lines1.reshape(-1, 3)
    # img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    # lines2 = cv2.computeCorrespondEpilines(
    #     pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
    # lines2 = lines2.reshape(-1, 3)
    # img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    # plt.subplot(121), plt.imshow(img5)
    # plt.subplot(122), plt.imshow(img3)
    # plt.suptitle("Epilines in both images")
    # plt.show()

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
    )

    return H1, H2, w1, h1, w2, h2

    # imgL = cv2.imread(r"rendered_train/Left/rgb_0044.PNG")
    # imgR = cv2.imread(r"rendered_train/Right/rgb_0044.PNG")

    # img1_r, img2_r = rectify(imgL, imgR)
    # draw_horizontal_lines(img1_r, img2_r)


def rectify_rendered_frame(img, H, w, h, save_path):
    img_name = re.split(r'[/.]+', img)[-2]
    img = cv2.imread(img)
    img_rectified = cv2.warpPerspective(img, H, (w, h))
    cv2.imwrite(os.path.join(save_path, "{}.PNG".format(img_name)), img_rectified)


if __name__ == "__main__":
    H1, H2, w1, h1, w2, h2 = main()

    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_path",
                        type=str,
                        default=r"rendered_train/")
    parser.add_argument("--output_path",
                        type=str,
                        default=r"rectified_rendered_data_uncalibrated/")

    args = parser.parse_args()

    create_dir(args.output_path)
    for view in ["Left", "Right"]:
        imgs_path = os.path.join(args.imgs_path, view)
        save_path = os.path.join(args.output_path, view)
        create_dir(save_path)
        print("Find rendered images in the {} directory ...".format(view))

        for i, img in enumerate(glob.glob(os.path.join(imgs_path, r"*.PNG"))):
            print("Start rectifying the rendered image --- {:04d}".format(i))
            if view == "Left":
                rectify_rendered_frame(img, H1, w1, h1, save_path)
            else:
                rectify_rendered_frame(img, H2, w2, h2, save_path)

    # draw_horizontal_lines(r"rectified_rendered_data_uncalibrated/Left/rgb_0531.PNG",
    #                       r"rectified_rendered_data_uncalibrated/Right/rgb_0531.PNG")
