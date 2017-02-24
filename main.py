import os
import numpy as np
import matplotlib as mpl

####################################################################################################
# Exercise 1: Power Iteration

def power_iteration(M: np.ndarray, epsilon: float = -1.0) -> (np.ndarray, list):
    """
    Compute largest eigenvector of matrix M using power iteration. It is assumed that the
    largest eigenvalue of M, in magnitude, is well separated.

    Arguments:
    M: matrix, assumed to have a well separated largest eigenvalue
    epsilon: epsilon used for convergence (default: 10 * machine precision)

    Return:
    vector: eigenvector associated with largest eigenvalue
    residuals : residual for each iteration step

    Raised Exceptions:
    ValueError: if matrix is not square

    Forbidden:
    numpy.linalg.eig, numpy.linalg.eigh, numpy.linalg.svd
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix not nxn")

    # set epsilon to default value if not set by user
    if epsilon < 0:
        epsilon = np.finfo(M.dtype).eps * 5

    # random vector of proper size to initialize iteration
    vector = np.random.randn(M.shape[0])
    vector /= np.linalg.norm(vector)

    # Initialize residual list and residual of current eigenvector estimate
    residuals = []
    residual = 2.0 * epsilon

    # Perform power iteration
    while residual > epsilon:
        tmp = np.dot(M, vector)
        tmp /= np.linalg.norm(tmp)
        residual = np.linalg.norm(tmp - vector)
        residuals.append(residual)
        vector = tmp

    return vector, residuals


####################################################################################################
# Exercise 2: Eigenfaces

def load_images(path: str, file_ending: str=".png") -> (list, int, int):
    """
    Load all images in path with matplotlib that have given file_ending

    Arguments:
    path: path of directory containing image files that can be assumed to have all the same dimensions
    file_ending: string that image files have to end with, if not->ignore file

    Return:
    images: list of images (each image as numpy.ndarray and dtype=float64)
    dimension_x: size of images in x direction
    dimension_y: size of images in y direction

    Raised Exceptions:
    IOError: if an image can't be loaded and converted to numpy.ndarray
    """

    paths = []
    images = []

    # read each image in path as numpy.ndarray and append to images
    for img in os.listdir(path):
        if img.endswith(file_ending):
            paths.append(img)
    paths.sort()

    for img in paths:
        images.append(np.asarray(mpl.image.imread(path + "/" + img), np.float64))

    # set dimensions according to first image in images
    dimension_y = images[0].shape[0]
    dimension_x = images[0].shape[1]

    return images, dimension_x, dimension_y


def setup_data_matrix(images: list) -> np.ndarray:
    """
    Create data matrix out of list of 2D data sets.

    Arguments:
    images: list of 2D images (assumed to be all homogeneous of the same size and type np.ndarray)

    Return:
    D: data matrix that contains the flattened images as rows
    """
    # initialize data matrix with proper size and data type
    D = np.zeros((len(images), images[0].size))

    # add flattened images to data matrix
    for i in range(0, len(images)):
        D[i] = images[i].flatten()

    return D


def calculate_pca(D: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Perform principal component analysis for given data matrix.

    Arguments:
    D: data matrix of size m x n where m is the number of observations and n the number of variables

    Return:
    pcs: matrix containing principal components as rows
    svals: singular values associated with principle components
    mean_data: mean that was subtracted from data
    """

    # subtract mean from data / center data at origin
    mean_data = np.mean(D, 0)
    D -= mean_data

    # compute left and right singular vectors and singular values
    u, svals, pcs = np.linalg.svd(D, full_matrices=False)

    return pcs, svals, mean_data


def accumulated_energy(singular_values: np.ndarray, threshold: float = 0.8) -> int:
    """
    Compute index k so that threshold percent of magnitude of singular values is contained in
    first k singular vectors.

    Arguments:
    singular_values: vector containing singular values
    threshold: threshold for determining k (default = 0.8)

    Return:
    k: threshold index
    """

    # Normalize singular value magnitudes
    singular_values /= np.sum(singular_values)

    # Determine k that first k singular values make up threshold percent of magnitude
    k = 0
    sum = 0.0
    for i in range(0, singular_values.shape[0]):
        sum += np.sum(singular_values[i])
        k += 1
        if sum > threshold:
            break

    return k


def project_faces(pcs: np.ndarray, images: list, mean_data: np.ndarray) -> np.ndarray:
    """
    Project given image set into basis.

    Arguments:
    pcs: matrix containing principal components / eigenfunctions as rows
    images: original input images from which pcs were created
    mean_data: mean data that was subtracted before computation of SVD/PCA

    Return:
    coefficients: basis function coefficients for input images, each row contains coefficients of one image
    """

    # initialize coefficients array with proper size
    coefficients = np.zeros((len(images), pcs.shape[0]))

    # iterate over images and project each normalized image into principal component basis
    for i in range(0, len(images)):
        coefficients[i] = np.dot(pcs, images[i].flatten()-mean_data)

    return coefficients


def identify_faces(coeffs_train: np.ndarray, pcs: np.ndarray, mean_data: np.ndarray, path_test: str) -> (
np.ndarray, list, np.ndarray):
    """
    Perform face recognition for test images assumed to contain faces.

    For each image coefficients in the test data set the closest match in the training data set is calculated.
    The distance between images is given by the angle between their coefficient vectors.

    Arguments:
    coeffs_train: coefficients for training images, each image is represented in a row
    path_test: path to test image data

    Return:
    scores: Matrix with correlation between all train and test images, train images in rows, test images in columns
    imgs_test: list of test images
    coeffs_test: Eigenface coefficient of test images
    """
    
    # TODO: load test data set
    imgs_test, x, y = load_images(path_test)

    # TODO: project test data set into eigenbasis
    coeffs_test =project_faces(pcs, imgs_test, mean_data)


    # TODO: Initialize scores matrix with proper size
    scores = np.zeros((len(coeffs_train),len(coeffs_test)))
    # TODO: Iterate over all images and calculate pairwise correlation
    for i in range(len(coeffs_test)):
        for j in range(len(coeffs_train)):
             scores[j,i]=np.math.acos((np.dot(coeffs_test[i], coeffs_train[j]))/(np.linalg.norm(coeffs_test[i])*np.linalg.norm(coeffs_train[j])))

    return scores, imgs_test, coeffs_test


if __name__ == '__main__':
    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
