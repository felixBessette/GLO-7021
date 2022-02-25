import math

import matplotlib.pyplot
from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from scipy.optimize import fmin
from numpy import unravel_index

CirclePixels = [(0, 3), (0, 4), (1, 5), (2, 6), (3, 6), (4, 6), (5, 5), (6, 4), (6, 3), (6, 2), (5, 1), (4, 0),
                (3, 0), (2, 0), (1, 1), (0, 2)]


def is_darker(seuil, pixel_intensity, center_intensity):
    return pixel_intensity < center_intensity - seuil


def is_brighter(seuil, pixel_intensity, center_intensity):
    return pixel_intensity > center_intensity + seuil


def get_criteria(seuil, image, center_intensity, cx, cy):
    darker = 0
    brighter = 0
    for i, j in [(0, 3), (3, 6), (6, 3), (3, 0)]:
        pixel_intensity = image[i + cx][j + cy]
        if is_darker(seuil, pixel_intensity, center_intensity):
            darker += 1
            if darker >= 3:
                return is_darker
        elif is_brighter(seuil, pixel_intensity, center_intensity):
            brighter += 1
            if brighter >= 3:
                return is_brighter

    return None


def pixel_test(image, center, pixel_position, score, criteria, seuil, center_intensity):
    x_offset, y_offset = pixel_position
    pixel_intensity = image[center[0] + x_offset][y_offset + center[1]]
    if criteria(seuil, pixel_intensity, center_intensity):
        score += pixel_intensity - center_intensity if pixel_intensity - center_intensity > seuil else center_intensity - pixel_intensity
        return True, score
    else:
        return False, score


def detection_coin_FAST(image, centre, seuil):
    cx, cy = tuple(centre)
    center_intensity = image[cx][cy]
    criteria = get_criteria(seuil, image, center_intensity, cx, cy)
    if criteria is None:
        return False, 0

    score, valid_pixels_count, pixel_start = 0, 0, 0
    for pixel_num, pixel_pos in enumerate(CirclePixels):
        is_valid, score = pixel_test(image, centre, pixel_pos, score, criteria, seuil, center_intensity)
        if is_valid:
            valid_pixels_count += 1
        else:
            if valid_pixels_count >= 12:
                if pixel_start == 0:
                    for k in range(len(CirclePixels) - 1, -1, -1):
                        is_valid, score = pixel_test(image, centre, pixel_pos, score, criteria, seuil, center_intensity)
                        if not is_valid:
                            break
                break
            valid_pixels_count = 0
            pixel_start = pixel_num + 1

    return valid_pixels_count >= 12, score


def test_detection_coin_FAST():
    img = cv2.imread('remise/source/bw-rectified-left-022148small.png')
    pixels = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    marked_pixels = np.zeros((len(pixels), len(pixels[0])))
    intensities = []
    corners_count = 0
    for i in range(8, len(pixels) - 8):
        for j in range(8, len(pixels[i]) - 8):
            is_corner, intensity = detection_coin_FAST(pixels, (i, j), 10.0)
            if is_corner:
                intensities.append(intensity)
                marked_pixels[i][j] = is_corner
                corners_count += 1

    plt.figure(2)
    plt.imshow(img)
    PositionDesCoins = np.argwhere(marked_pixels > 0)
    x, y = PositionDesCoins.T
    x = x
    y = y
    plt.scatter(y, x, marker='.', facecolors='none', edgecolors='r')

    matplotlib.pyplot.tight_layout()
    plt.show()
    plt.hist(intensities)
    plt.show()
    corner_percentage = corners_count / (len(pixels) * len(pixels[0]))
    print("Nombre total de coins" + str(corners_count))
    print("Pourcentage des pixels considérés comme des coins = " + str(corner_percentage * 100))


def ExtractBRIEF(ImagePatch, BriefDescriptorConfig):
    descriptor = []
    for a, b in BriefDescriptorConfig:
        ax, ay = a
        bx, by = b
        descriptor.append(ImagePatch[ax][ay] > ImagePatch[bx][by])
    return np.array(descriptor)


def Extract_Features(image, briefDescriptorConfig):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners_count = 0
    marked_pixels = np.zeros(image.shape, dtype=float)
    for x in range(8, len(image) - 8):
        for y in range(8, len(image[x]) - 8):
            is_corner, intensity = detection_coin_FAST(image, (x, y), 10.0)
            if is_corner:
                corners_count += 1
                marked_pixels[x][y] = intensity

    local_max_intensities = []
    best_corners = []
    for x in range(8, marked_pixels.shape[0] - 8, 15):
        for y in range(8, marked_pixels.shape[1] - 8, 15):
            patch = marked_pixels[x:x + 15, y:y + 15]
            x_offset, y_offset = unravel_index(patch.argmax(), patch.shape)
            local_max_x = x + x_offset
            local_max_y = y + y_offset
            local_max_intensity = marked_pixels[local_max_x][local_max_y]
            if local_max_intensity > 0:
                local_max_intensities.append(marked_pixels[local_max_x][local_max_y])
                best_corners.append((local_max_x, local_max_y))
    best_corners_v2 = []
    local_max_intensity_v2 = []
    for i in range(8, marked_pixels.shape[0] - 8):
        for j in range(8, marked_pixels.shape[1] - 8):
            intensity = marked_pixels[i][j]
            if intensity <= 0:
                continue
            patch = marked_pixels[i - 7:i + 7, j - 7:j + 7]
            if np.max(patch) <= intensity:
                local_max_intensity_v2.append(intensity)
                best_corners_v2.append((i, j))
    best_corners = np.array(best_corners_v2)
    local_max_intensities = np.array(local_max_intensity_v2)
    percentage_removed = str(100 - ((100 * len(best_corners)) / corners_count))
    print("Suppression des non-maxima locaux a retiré : " + percentage_removed + "% des coins")

    sorted_intensities = np.argsort(local_max_intensities)
    percentile90_index = int(np.ceil(90 * len(sorted_intensities) / 100))
    corner_index_to_keep = sorted_intensities[percentile90_index:]
    cornersToKeep = best_corners[corner_index_to_keep]

    percentage_removed = str(100 - (100 * len(cornersToKeep) / len(best_corners)))
    print("Avec sélection des coins, nous avons : " + percentage_removed + "% des coins initiaux retirés")

    plt.imshow(image)
    y = cornersToKeep[:, 1]
    x = cornersToKeep[:, 0]
    plt.scatter(y, x, marker='.', facecolors='none', edgecolors='r')
    plt.show()
    descriptors = []
    for x_offset, y_offset in cornersToKeep:
        imagePatch = image[x_offset - 8:x_offset + 7, y_offset - 8: y_offset + 7]
        descriptors.append((x_offset, y_offset, ExtractBRIEF(imagePatch, briefDescriptorConfig)))
    return descriptors


def hamming_distance(descriptorL, descriptorR):
    return np.sum(descriptorL != descriptorR)


def compare_images():
    briefDescriptorConfig = []
    for x in range(200):
        a = (random.randint(0, 14), random.randint(0, 14))
        b = (random.randint(0, 14), random.randint(0, 14))
        briefDescriptorConfig.append((a, b))

    imgRight = cv2.imread('remise/source/bw-rectified-right-022148small.png')
    descriptorsRight = Extract_Features(imgRight, briefDescriptorConfig)
    imgLeft = cv2.imread('remise/source/bw-rectified-left-022148small.png')
    descriptorsLeft = Extract_Features(imgLeft, briefDescriptorConfig)
    plt.figure(2)
    plt.imshow(imgLeft)
    show_em = True
    for x_L, y_L, descriptor_L in descriptorsLeft:
        min_distance = math.inf
        min_corner = None
        for x_R, y_R, descriptor_R in descriptorsRight:
            if show_em:
                plt.plot([y_L, y_R], [x_L, x_R], color="green", lw=1)
            distance = hamming_distance(descriptor_L, descriptor_R)
            if min_distance > distance:
                min_corner = (x_R, y_R)
                min_distance = distance

        plt.plot([y_L, min_corner[1]], [x_L, min_corner[0]], color="red", lw=1)
        matplotlib.pyplot.tight_layout()
        if show_em:
            plt.show()
            plt.imshow(imgLeft)
        show_em = False

    matplotlib.pyplot.tight_layout()
    plt.show()


# img = cv2.imread('remise/source/bw-rectified-right-022148small.png')
# pixels = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
test_detection_coin_FAST()


def reprojection(H, focale, L):
    """
    Calculates the position of the 3 points L in respect to the image camera plan coordinates u, v
    :param H: camera position expressed by a 4x4 matrix
    :param focale: focal distance of the camera
    :param L: 3x3 matrix where each row is a point which we project
    :return: position of the 3 points L in respect to the image camera plan coordinates u and v
    |3  5   3|1 0 0|
    |2  6  2|0 1 0|
    |3  4  1|0 0 1|
    """
    intrinsic_matrix = np.zeros((4, 4))
    np.fill_diagonal(intrinsic_matrix, 1)
    intrinsic_matrix[2][2] = 1 / focale

    extrinsic_matrix = np.linalg.inv(H)
    Lc = []
    for Li in L:
        Lci = intrinsic_matrix @ extrinsic_matrix @ np.append(Li, 1)
        Lci = Lci[:-1] / Lci[-1]
        Lc.append(Lci)
    return np.array(Lc)


def compute_target_image():
    L = np.array([np.array([-0.2, 0, 1.2]), np.array([0, 0, 1]), np.array([0.2, 0, 1.2])])
    H = np.array([np.array([1, 0, 0, 0]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]), np.array([0, 0, 0, 1])])
    C = reprojection(H, 1000, L)
    return C


def somme_des_residuels_au_carre(pose_camera, focal, L, C):
    result = 0
    x, z, theta = tuple(pose_camera)
    H = [
        [math.cos(theta), 0, math.sin(theta), x],
        [0, 1, 0, 0],
        [-math.sin(theta), 0, math.cos(theta), z],
        [0, 0, 0, 1]
    ]

    for Li, Ci in zip(L, C):
        result += np.linalg.norm(Ci - reprojection(H, focal, [Li])) ** 2

    return result


def minimize_reprojection_error():
    L = np.array([np.array([-0.2, 0, 1.2]), np.array([0, 0, 1]), np.array([0.2, 0, 1.2])])
    pose_initiale_camera = np.array([.2, .2, .2])
    C = compute_target_image()
    pose_solution = fmin(somme_des_residuels_au_carre, pose_initiale_camera,
                         args=(1000, L, C), maxiter=1000)
    return pose_solution


def noise_impact():
    L = np.array([np.array([-0.2, 0, 1.2]), np.array([0, 0, 1]), np.array([0.2, 0, 1.2])])
    camera_pose = np.array([.2, .2, .2])
    C = compute_target_image()
    x_coordinates = []
    y_coordinates = []
    for _ in range(7):
        camera_pose[1] -= 1
        for _ in range(1000):
            C_noised = C.copy()
            for Ci in C_noised:
                Ci[0] += np.random.normal(0, 2, 1)[0]
            camera_pose = fmin(somme_des_residuels_au_carre, camera_pose, args=(1000, L, C_noised), maxiter=1000)
            x_coordinates.append(camera_pose[0])
            y_coordinates.append(camera_pose[1])
        plt.scatter(x_coordinates, y_coordinates)
        plt.show()
