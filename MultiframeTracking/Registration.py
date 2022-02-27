import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline
from sklearn.neighbors import NearestNeighbors


######################################################
# Helper Methods
######################################################

def get_differential_filter():
    """Return differential filters along x and y"""
    filter_x = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]
    filter_y = [[-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]]
    filter_x, filter_y = np.array(filter_x), np.array(filter_y)
    return filter_x, filter_y


def filter_image(im, filter):
    """Pad and filter given image"""

    def zero_pad_image(im, padding_x, padding_y):
        """Adds zero padding for the specified amount"""
        padded_im = np.zeros((im.shape[0] + 2 * padding_y, im.shape[1] + 2 * padding_x))
        padded_im[padding_y:-padding_y, padding_x:-padding_x] = im
        return padded_im

    im_padded = zero_pad_image(im, 1, 1)

    im_filtered = np.zeros(im.shape)
    filter_size_y, filter_size_x = filter.shape[0], filter.shape[1]

    for j in range(im.shape[0]):
        for i in range(im.shape[1]):
            im_filtered[j][i] = np.sum(im_padded[j:(j + filter_size_y), i:(i + filter_size_x)] * filter)

    return np.array(im_filtered)


def interpolate(x_prime, img, output_shape):
    """
    Perform Bi-linear interpolation
    :param x_prime: flattened array of indices
    :param img: image
    :param output_shape: the shape of the output
    :return: output image with interpolated values
    """
    x, y = x_prime[0, :], x_prime[1, :]

    # calculate points before and after in the integer image grid
    x_floor = np.clip(np.floor(x).astype(int), 0, img.shape[1] - 1)
    x_ceil = np.clip(x_floor + 1, 0, img.shape[1] - 1)
    y_floor = np.clip(np.floor(y).astype(int), 0, img.shape[0] - 1)
    y_ceil = np.clip(y_floor + 1, 0, img.shape[0] - 1)

    # get image values at the floors/ceilings for all the points
    I11 = img[y_floor, x_floor]
    I12 = img[y_ceil, x_floor]
    I21 = img[y_floor, x_ceil]
    I22 = img[y_ceil, x_ceil]

    # assign a weight to each distance for all the points
    # (the denominator for the weighted equation has (x_ceil - x_floor)*(y_ceil - y_floor), which is just 1)
    w11 = (x_ceil - x) * (y_ceil - y)
    w12 = (x_ceil - x) * (y - y_floor)
    w21 = (x - x_floor) * (y_ceil - y)
    w22 = (x - x_floor) * (y - y_floor)

    # return interpolated image in the shape of the output
    return (w11 * I11 + w12 * I12 + w21 * I21 + w22 * I22).reshape(output_shape)


######################################################
# Main Functions
######################################################

def find_match(img1, img2):
    """
    Use SIFT to extract features, filter using a ratio test and perform
    bi-directional consistency checks
    ----
    :param img1: the first image
    :param img2: the second image
    :return: the best key points in img1 and its set of corresponding points
    """

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # get key points
    kp1_pts = np.array([kp1[k].pt for k in range(len(kp1))])
    kp2_pts = np.array([kp2[k].pt for k in range(len(kp2))])

    # (optional) [for the plots in the report]
    # visualize_sift_features(kp1, kp2, img1, img2)

    # use k-NN to find the two closest neighbors
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(des2)
    lr_dist, lr_idx = knn.kneighbors(des1)
    knn.fit(des1)
    rl_dist, rl_idx = knn.kneighbors(des2)

    # filter according to the ratio test for both images
    ratio1 = np.divide(lr_dist[:, 0], lr_dist[:, 1]) < 0.75
    best_kp1, best_des1 = kp1_pts[ratio1], lr_idx[ratio1][:, 0]
    best_corresponding_points1 = np.array([kp2[b1].pt for b1 in best_des1])

    ratio2 = np.divide(rl_dist[:, 0], rl_dist[:, 1]) < 0.75
    best_kp2, best_des2 = kp2_pts[ratio2], rl_idx[ratio2][:, 0]
    best_corresponding_points2 = np.array([kp1[b2].pt for b2 in best_des2])

    # check for bi-directional consistency - store the indices of points to retain in lr_consistency
    lr_consistency = []
    for b in range(len(best_kp1)):
        for c in range(len(best_kp2)):
            if (best_corresponding_points1[b].all() == best_kp2[c].all()) and (
                    best_corresponding_points2[c].all() == best_kp1[b].all()):
                lr_consistency.append(True)
                break
            else:
                lr_consistency.append(False)
    return best_kp1[lr_consistency], best_corresponding_points1[lr_consistency]


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    """
    Align an image by finding the best Affine transform using RANSAC
    :param x1: key points set 1
    :param x2: key points set 2
    :param ransac_thr: RANSAC threshold for error
    :param ransac_iter: RANSAC iterations
    :return: best Affine transform found
    """

    np.random.seed(42)

    # number of points to sample
    sample_points = 3

    # placeholders
    A = np.zeros((6, 6))
    A_best = np.zeros((6, 6))
    max_inliers = -1

    for r in range(ransac_iter):
        pt_idx = np.random.choice(x1.shape[0], sample_points, replace=False)  # sample without replacing
        x1_sampled, x2_sampled = x1[pt_idx], x2[pt_idx]

        # create matrix A in Ax = b
        x1_pad_1 = np.c_[x1_sampled, np.ones((3, 1))]
        x1_mat = np.zeros((6, 6))
        x1_pad_zeros = np.c_[x1_pad_1, np.zeros((3, 3))]
        x1_pad_zeros_flipped = np.c_[np.zeros((3, 3)), x1_pad_1]
        for c in range(x1_mat.shape[0]):
            if c % 2 == 0:
                x1_mat[c] = x1_pad_zeros[c // 2]
            else:
                x1_mat[c] = x1_pad_zeros_flipped[c // 2]

        # create matrix b in Ax = b
        x2_mat = np.hstack(x2_sampled).T

        # get the Affine transform x in Ax = b
        A = (np.linalg.inv(x1_mat.T @ x1_mat) @ x1_mat.T @ x2_mat).reshape(2, 3)
        A = np.concatenate((A, [[0, 0, 1]]), axis=0)

        # get transformed points and store error
        x2_preds = A @ np.c_[x1, np.ones((x1.shape[0], 1))].T
        error_list = np.linalg.norm(x2_preds[:2, :].T - x2, axis=1)

        # count inliers, store the best Affine transform
        inliers = np.count_nonzero(np.where(error_list < ransac_thr))
        if inliers > max_inliers:
            max_inliers = inliers
            A_best = A.copy()

    return A_best


def warp_image(img, A, output_size):
    """
    Warps the given image using inverse mapping
    :param img: image
    :param A: Affine transform
    :param output_size: size of the output image
    :return: interpolated output image
    """

    # we create an index matrix, that contains the indices for each pixel in the output
    xy_idx = np.indices(output_size).reshape(2, -1)
    xy_idx[[0, 1], :] = xy_idx[[1, 0], :]
    idx_matrix = np.concatenate((xy_idx, np.ones((1, output_size[0] * output_size[1]))), axis=0)

    # using the Affine transform, we warp each location in the target image (with our index) to the template image
    x_inv = np.einsum('ij, jn -> in', A, idx_matrix)

    # interpolation of points
    interp_img = interpolate(x_inv, img, output_size)
    return interp_img

    # using RectBivariateSpline from SciPy also works -- slower
    # xi, yi = np.arange(img.shape[0]), np.arange(img.shape[1])
    # rbs = RectBivariateSpline(xi, yi, img)
    # img_warped = rbs.ev(x_inv[1, :], x_inv[0, :]).reshape(output_size)
    # return img_warped


def align_image(template, target, A):
    """
    Aligns a template image to the target by refining the Affine transformation
    using Inverse Compositional Alignment
    :param template: template image
    :param target: target image
    :param A: initial Affine transform
    :return: refined Affine transform
    """

    def get_affine(p_list):
        """
        Get Affine transformation from the given parameters
        :param p_list: Parameter list
        :return: Affine Transformation
        """
        return np.array([[p_list[0][0] + 1, p_list[1][0], p_list[2][0]],
                         [p_list[3][0], p_list[4][0] + 1, p_list[5][0]],
                         [0, 0, 1]])

    # normalize
    template = template / 255.
    target = target / 255.

    # calculate gradients for template and stack
    diff_x, diff_y = get_differential_filter()
    tpl_grad_x = filter_image(template, diff_x)
    tpl_grad_y = filter_image(template, diff_y)
    grad = np.stack((tpl_grad_x, tpl_grad_y))
    grad = grad.reshape((2, -1))

    # create indices matrix for calculating Jacobian for all points
    # each [ u1 v1 1 0 0 0 ] is from idx_matrix
    # each [ 0 0 0 u1 v1 1 ] is from idx_matrix_flip
    xy_idx = np.indices(template.shape).reshape(2, -1)
    xy_idx[[0, 1], :] = xy_idx[[1, 0], :]
    idx_matrix = np.concatenate((xy_idx.T,
                                 np.ones((template.shape[0] * template.shape[1], 1)),
                                 np.zeros((template.shape[0] * template.shape[1], 3))), axis=1)
    idx_matrix_flip = np.concatenate((np.zeros((template.shape[0] * template.shape[1], 3)),
                                      xy_idx.T,
                                      np.ones((template.shape[0] * template.shape[1], 1)),), axis=1)
    # allocate space and values to Jacobian
    mega_jacobian = np.empty((2 * idx_matrix.shape[0], idx_matrix.shape[1]), dtype=idx_matrix.dtype)
    mega_jacobian[0::2] = idx_matrix
    mega_jacobian[1::2] = idx_matrix_flip
    mega_jacobian = np.concatenate(mega_jacobian, axis=0).reshape((-1, 2, 6))
    del idx_matrix, idx_matrix_flip  # delete to save memory

    # calculate steepest descent images using gradient and Jacobian
    sd_images = np.einsum('ni, nij -> nj', grad.T, mega_jacobian)

    # calculate Hessian from steepest descent images, and find its inverse
    H = np.einsum('ab, bc -> ac', sd_images.T, sd_images)
    H_inv = np.linalg.inv(H)

    # reshape for easier error calculation
    sd_images = sd_images.reshape((452, 292, 6))
    # plot steepest descent images if needed
    # sd_img = sd_images[:, :, 0]
    # for i in range(1, 6):
    #     temp = np.hstack((sd_img, sd_images[:, :, i]))
    # plt.imshow(sd_img, cmap='gray')
    # plt.axis('off')
    # plt.show()

    A_refined = A.copy()
    errors_list = []

    max_epochs = 10000
    # iterate until convergence / max iteration limit is reached
    for epoch in range(max_epochs):
        # warp target
        target_warped = warp_image(target, A_refined, template.shape)

        # calculate target - template error and its norm
        I_error = (target_warped - template).reshape((template.shape[0], template.shape[1], 1))
        I_error_norm = np.linalg.norm(I_error)
        errors_list.append(I_error_norm)

        # calculate F
        F = np.einsum('ijk, ijl -> kl', sd_images, I_error).reshape(6, 1)

        # calculate delta_p and its norm
        delta_p = H_inv @ F
        error_p = np.linalg.norm(delta_p)

        # refine A
        A_delta_p = np.linalg.inv(get_affine(delta_p))
        A_refined = A_refined @ A_delta_p

        # convergence criterion
        if error_p < 1e-3:
            break
    print('Alignment finished!')
    return A_refined, np.array(errors_list)


def track_multi_frames(template, img_list):
    """
    Track template across multiple images
    :param template: template image to track
    :param img_list: list of images to track template image in
    :return: list of Affine transformations from one image in the target list to the next
    """
    # SIFT Feature Matching
    x1, x2 = find_match(template, target_list[0])

    # RANSAC to get initial A
    ransac_threshold = 4
    ransac_iterations = 10000
    A = align_image_using_feature(x1, x2, ransac_threshold, ransac_iterations)

    # align and warp loop
    Affine_list = []
    image_count = 0
    for img in img_list:
        print('Current image: ', image_count + 1)
        image_count += 1
        A_refined, _ = align_image(template, img, A)
        Affine_list.append(A_refined)
        template = warp_image(img, A_refined, template.shape)
        A = A_refined.copy()

    return Affine_list


######################################################
# Visualization Functions
######################################################

def visualize_sift_features(keypoints1, keypoints2, img1, img2, img_h=500):
    """Draw SIFT Features from given key points for two images"""
    sift_1 = cv2.drawKeypoints(img1, keypoints1, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    sift_2 = cv2.drawKeypoints(img2, keypoints2, img2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    scale_factor1 = img_h / sift_1.shape[0]
    scale_factor2 = img_h / sift_2.shape[0]
    img1_resized = cv2.resize(sift_1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(sift_2, None, fx=scale_factor2, fy=scale_factor2)
    img = np.hstack((img1_resized, img2_resized))
    plt.rcParams['figure.dpi'] = 240
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()


def visualize_bounding_box(img1, img2, x1, x2, A, img_h=500, ransac_thr=4):
    """Draw the bounding box and highlight the inliers"""
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    borders = np.array([[0, 0, 1],
                        [0, img1.shape[0] - 1, 1],
                        [img1.shape[1] - 1, img1.shape[0] - 1, 1],
                        [img1.shape[1] - 1, 0, 1],
                        [0, 0, 1]])
    transformed_borders = np.array(A @ borders.T)[:2, :].T
    x2_preds = A @ np.c_[x1, np.ones((x1.shape[0], 1))].T
    inliers = np.where(np.linalg.norm(x2_preds[:2, :].T - x2, axis=1) < ransac_thr)[0]
    scale_factor1 = img_h / img1.shape[0]
    scale_factor2 = img_h / img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    transformed_borders *= scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    transformed_borders[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        if i in inliers:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'g')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'go', fillstyle='none')
        else:
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
            plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo', fillstyle='none')
    for i in range(1, transformed_borders.shape[0]):
        plt.plot([transformed_borders[i - 1, 0], transformed_borders[i, 0]],
                 [transformed_borders[i - 1][1], transformed_borders[i][1]], 'r-')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h / img1.shape[0]
    scale_factor2 = img_h / img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.rcParams['figure.dpi'] = 240
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo', fillstyle='none')
    plt.axis('off')
    plt.show()


def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                          [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.rcParams['figure.dpi'] = 480
    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i + 1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 4
    ransac_iter = 10000
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)
    visualize_bounding_box(template, target_list[0], x1, x2, A)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)
