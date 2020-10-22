import numpy as np
from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
import cv2
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
import os
import matplotlib.pyplot as plt
import PIL
from PIL import  Image
float_tolerance = 1e-7


def findscaleExtrema(pt, gaussian_images, dog_images, num_intervals, sigma, image_border_width, contrast_threshold=0.04):
    x = pt[0] # punto lungo le righe
    y = pt[1] # punto lungo le colonne

    scalepoints = []
    for octave_index,dog_images_in_octave in enumerate(dog_images):
        dog_values = []
        for ii,imgs in enumerate(dog_images_in_octave):
            values = imgs[x//2,y//2]
            dog_values.append(values)
        scalepoints.append(dog_values)

    scalepoints = np.array(scalepoints)
    t = np.argmax(scalepoints)
    oct_index = t//scalepoints.shape[1]

    gaussian_index = t%scalepoints.shape[1]
    maximo = (oct_index,gaussian_index)
    kpt= KeyPoint()
    kpt.pt = (y*(2**oct_index),x*(2**oct_index))
    kpt.response = np.abs(np.max(scalepoints))
    kpt.size = sigma * (2**((gaussian_index)/np.float32(num_intervals))) * (2 **(oct_index + 1))
    kpt.octave = int(oct_index) + int(gaussian_index) * (2 ** 8) + int(round((0 + 0.5) * 255)) * (2 ** 16)
    return kpt, gaussian_index


def computekeypointwithorientation(keypoint, octave_index, gaussian_image, radius_factor = 3, num_bins = 36, peak_ratio = 0.8, scale_factor = 1.5):
    image_shape = gaussian_image.shape
    keypoints_with_orientations = []
    new_keypoint = None
    # compute scale
    scale = scale_factor * keypoint.size / np.float32(2**(octave_index + 1))
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)
    for i in range(-radius , radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < image_shape[0] - 1:
            for j in range(-radius, radius, + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y,region_x + 1 ] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1,region_x  ] - gaussian_image[region_y + 1, region_x ]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy,dx))
                    #print("***********")
                    #print(dx)
                    #print(dy)
                    #print(gradient_magnitude)
                    #print(gradient_orientation)
                    weight = np.exp(weight_factor * (i**2 + j **2))
                    #prin
                    histogram_index = int(np.round(gradient_orientation * num_bins/360.))

                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
    orientation_max = max(smooth_histogram)
    orientation_peaks = where(logical_and(smooth_histogram > roll(smooth_histogram, 1), smooth_histogram > roll(smooth_histogram, -1)))[0]
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30) in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < float_tolerance:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    return keypoints_with_orientations





def convertKeypointsToInputImageSize(keypoints):
    """Convert keypoint point, size, and octave to input image size
    """
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints






def unpackOctave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale
