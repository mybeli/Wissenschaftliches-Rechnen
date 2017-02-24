
import numpy as np
import matplotlib.pyplot as plt
import unittest
from main import power_iteration, load_images, setup_data_matrix, calculate_pca, accumulated_energy, project_faces, \
    identify_faces
from lib import visualize_eigenfaces, plot_singular_values_and_energy, plot_identified_faces


class Tests(unittest.TestCase):
    def test_0_power_iteration(self):
        T = np.array([[0.5, 0.25, 0.25], [0.5, 0.0, 0.5], [0.25, 0.25, 0.5]])
        evector, residuals = power_iteration(T.transpose())
        evalues_ref, evectors_ref = np.linalg.eig(T.transpose())
        self.assertTrue(np.abs(evalues_ref[0] - 1.0) < 10.0 * np.finfo(T.dtype).eps)
        product = np.dot(evector, evectors_ref[:, 0])
        self.assertTrue(np.isclose(np.abs(product), 1.0))  # Check for right direction
        self.assertTrue(np.isclose(np.linalg.norm(evector), 1.0))  # Check for unit length

        # plot convergence behavior
        plt.plot(residuals, '-rx')
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Number of Iteration')
        plt.ylabel('Estimated Error')
        plt.legend(['T'])
        plt.show()

    def setup_tests(self, stage, cutoff_threshold=0.8):
        # Read training data set
        self.imgs_train, self.dim_x, self.dim_y = load_images("./data/train/")
        if stage == "load_images":
            return

        # compute data matrix
        self.D = setup_data_matrix(self.imgs_train)
        if stage == "setup_data_matrix":
            return

        # Perform principal component analysis
        self.pcs, self.sv, self.mean_data = calculate_pca(self.D)
        if stage == "calculate_pca":
            return

        # compute threshold for 90% of spectral energy
        self.k = accumulated_energy(self.sv, cutoff_threshold)
        if stage == "accumulated_energy":
            return

        # cut off number of pcs if desired
        self.pcs = self.pcs[0:self.k, :]
        # compute coefficients of input in eigenbasis
        self.coeffs_train = project_faces(self.pcs, self.imgs_train, self.mean_data)
        if stage == "project_faces":
            return

        # perform classical face recognition
        self.scores, self.imgs_test, self.coeffs_test = identify_faces(self.coeffs_train, self.pcs, self.mean_data,
                                                                       './data/test/')

    def test_1_load_images(self):
        self.setup_tests("load_images")
        # check if images are loaded properly
        self.assertTrue(len(self.imgs_train) == 150)
        self.assertTrue(isinstance(self.imgs_train[0], np.ndarray))
        self.assertTrue(
            self.dim_x == self.imgs_train[0].shape[1] == 98 and self.dim_y == self.imgs_train[0].shape[0] == 116)

    def test_2_setup_data_matrix(self):
        self.setup_tests("setup_data_matrix")
        self.assertTrue(isinstance(self.D, np.ndarray))
        self.assertTrue(self.D.shape[0] == len(self.imgs_train) and self.D.shape[1] == self.dim_x * self.dim_y == 11368)

    def test_3_calculate_pca(self):
        self.setup_tests("calculate_pca")
        # Test certain properties of the pca
        self.assertTrue(
            isinstance(self.pcs, np.ndarray) and isinstance(self.sv, np.ndarray) and isinstance(self.mean_data,
                                                                                                np.ndarray))
        self.assertTrue(self.pcs.shape[0] == len(self.imgs_train) == 150)
        self.assertTrue(self.pcs.shape[1] == self.dim_x * self.dim_y == 11368)
        self.assertTrue(self.mean_data.shape[0] == 11368)

        # Visualize the eigenfaces/principal components
        visualize_eigenfaces(10, self.pcs, self.sv, self.dim_x, self.dim_y)

    def test_4_accumulated_energy(self):
        self.setup_tests("accumulated_energy")
        self.assertTrue(self.k > 0)
        plot_singular_values_and_energy(self.sv, self.k)

    def test_5_project_faces(self):
        self.setup_tests("project_faces")
        self.assertTrue(self.coeffs_train.shape == (len(self.imgs_train), self.pcs.shape[0]))

    def test_6_identify_faces(self):
        self.setup_tests("identify_faces")
        self.assertTrue(self.scores.shape == (self.coeffs_train.shape[0], self.coeffs_test.shape[0]) != (1, 1))

        plot_identified_faces(self.scores, self.imgs_train, self.imgs_test, self.pcs, self.coeffs_test, self.mean_data)


if __name__ == '__main__':
    unittest.main()

