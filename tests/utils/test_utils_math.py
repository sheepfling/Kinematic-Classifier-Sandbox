from __future__ import annotations

import unittest

from kinematic_classifier_sandbox.utils.math import (
    centroid,
    clamp,
    cluster_balance,
    cluster_purity,
    covariance_matrix,
    dot_product,
    euclidean_distance,
    farthest_first_initialization,
    gaussian_logpdf,
    kmeans,
    linear_fit,
    logsumexp,
    matrix_deflation,
    matrix_vector_multiply,
    mean,
    median3,
    normalize_log_scores,
    normalize_vector,
    outer_product,
    power_iteration,
    project_rows,
    reconstruct_rows,
    safe_log,
    silhouette_score,
    std,
    vector_norm,
)


class UtilsMathTests(unittest.TestCase):
    def test_basic_scalar_helpers(self) -> None:
        self.assertEqual(clamp(3.0, 0.0, 2.0), 2.0)
        self.assertEqual(mean([1.0, 2.0, 3.0]), 2.0)
        self.assertAlmostEqual(std([1.0, 2.0, 3.0]), 1.0)
        self.assertAlmostEqual(safe_log(1.0), 0.0)
        self.assertEqual(median3([9.0, 1.0, 5.0], 1), 5.0)

    def test_log_helpers(self) -> None:
        self.assertAlmostEqual(logsumexp([0.0, 0.0]), 0.6931471805599453)
        normalized = normalize_log_scores({"a": 0.0, "b": 0.0})
        self.assertAlmostEqual(normalized["a"], 0.5)
        self.assertAlmostEqual(normalized["b"], 0.5)
        self.assertLess(gaussian_logpdf(0.0, 0.0, 1.0), 0.0)

    def test_linear_fit(self) -> None:
        intercept, slope = linear_fit([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
        self.assertAlmostEqual(intercept, 1.0)
        self.assertAlmostEqual(slope, 2.0)

    def test_vector_and_matrix_ops(self) -> None:
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(dot_product(v1, v2), 32.0)
        self.assertAlmostEqual(vector_norm([3.0, 4.0]), 5.0)
        
        normalized = normalize_vector([3.0, 4.0])
        self.assertAlmostEqual(normalized[0], 0.6)
        self.assertAlmostEqual(normalized[1], 0.8)
        
        # Zero vector normalization
        self.assertEqual(normalize_vector([0.0, 0.0]), [0.0, 0.0])
        
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        vec = [5.0, 6.0]
        self.assertEqual(matrix_vector_multiply(matrix, vec), [17.0, 39.0])
        
        outer = outer_product([1.0, 2.0])
        self.assertEqual(outer, [[1.0, 2.0], [2.0, 4.0]])
        
        deflated = matrix_deflation([[10.0, 0.0], [0.0, 5.0]], 10.0, [1.0, 0.0])
        self.assertEqual(deflated, [[0.0, 0.0], [0.0, 5.0]])

    def test_statistical_ops(self) -> None:
        data = [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 3.0]
        ]
        # means are [2.0, 2.0]
        # centered: [-1, 0], [0, -1], [1, 1]
        # covariance:
        # [ (-1*-1 + 0*0 + 1*1)/2, (-1*0 + 0*-1 + 1*1)/2 ]
        # [ (0*-1 + -1*0 + 1*1)/2, (0*0 + -1*-1 + 1*1)/2 ]
        # = [ [1.0, 0.5], [0.5, 1.0] ]
        cov = covariance_matrix(data)
        self.assertAlmostEqual(cov[0][0], 1.0)
        self.assertAlmostEqual(cov[0][1], 0.5)
        self.assertAlmostEqual(cov[1][0], 0.5)
        self.assertAlmostEqual(cov[1][1], 1.0)

    def test_power_iteration(self) -> None:
        # Simple diagonal matrix, largest eigenvalue is 10.0
        matrix = [[10.0, 0.0], [0.0, 5.0]]
        eigenvalue, eigenvector = power_iteration(matrix)
        self.assertAlmostEqual(eigenvalue, 10.0)
        self.assertAlmostEqual(abs(eigenvector[0]), 1.0, places=4)
        self.assertAlmostEqual(eigenvector[1], 0.0, places=4)

    def test_advanced_matrix_and_clustering(self) -> None:
        rows = [[1.0, 0.0], [0.0, 1.0], [1.1, 0.1]]
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        
        # euclidean
        self.assertAlmostEqual(euclidean_distance([0, 0], [3, 4]), 5.0)
        
        # project/reconstruct
        projected = project_rows(rows, vectors, 1)
        self.assertEqual(len(projected), 3)
        self.assertEqual(len(projected[0]), 1)
        self.assertAlmostEqual(projected[0][0], 1.0)
        
        reconstructed = reconstruct_rows(projected, vectors, 1)
        self.assertEqual(len(reconstructed), 3)
        self.assertEqual(len(reconstructed[0]), 2)
        self.assertAlmostEqual(reconstructed[0][0], 1.0)
        self.assertAlmostEqual(reconstructed[0][1], 0.0)
        
        # row_mean / centroid
        c = centroid(rows)
        self.assertAlmostEqual(c[0], (1.0 + 0.0 + 1.1) / 3)
        self.assertAlmostEqual(c[1], (0.0 + 1.0 + 0.1) / 3)
        
        # kmeans / initialization
        # With 2 points far apart, farthest first should pick them
        points = [[0.0, 0.0], [10.0, 10.0], [0.1, 0.1]]
        init = farthest_first_initialization(points, 2)
        self.assertEqual(len(init), 2)
        self.assertIn([0.0, 0.0], init)
        self.assertIn([10.0, 10.0], init)
        
        labels, centers, inertia = kmeans(points, 2)
        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(len(centers), 2)
        self.assertLess(inertia, 1.0)
        
        # silhouette
        score = silhouette_score(points, labels)
        self.assertGreater(score, 0.5)
        
        # purity / balance
        # labels for points: [0, 1, 0] or [1, 0, 1]
        truth = ["A", "B", "A"]
        # If labels match truth perfectly
        p = cluster_purity([0, 1, 0], truth)
        self.assertAlmostEqual(p, 1.0)
        
        b = cluster_balance([0, 1, 0])
        self.assertAlmostEqual(b, 0.5) # 1/2


if __name__ == "__main__":
    unittest.main()
