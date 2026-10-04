"""Regression checks for statistical values and lecture plotting layouts."""

import unittest

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

import utils


class ModernEnvironmentTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_pointwise_waic_known_probabilities(self):
        likelihood = xr.Dataset({
            "y": (("chain", "draw", "observation"), np.log([[[0.2, 0.8], [0.8, 0.2]]]))
        })
        trace = xr.DataTree.from_dict({"log_likelihood": likelihood})
        expected = np.log(0.5) - (np.log(4) / 2) ** 2
        np.testing.assert_allclose(utils.pointwise_waic(trace), [expected, expected])

    def test_waic_requires_selection_for_multiple_likelihoods(self):
        likelihood = xr.Dataset({
            name: (("chain", "draw", "observation"), -np.ones((2, 10, 3)))
            for name in ("a", "b")
        })
        trace = xr.DataTree.from_dict({"log_likelihood": likelihood})
        with self.assertRaises(ValueError):
            utils.pointwise_waic(trace)
        np.testing.assert_allclose(utils.pointwise_waic(trace, "b"), -np.ones(3))

    def test_density_preserves_axis_and_normalization(self):
        _, ax = plt.subplots()
        result = utils.plot_density(np.linspace(-1, 1, 1000), bw=0.1, ax=ax)
        self.assertIs(result, ax)
        xs, density = ax.get_lines()[0].get_data()
        self.assertAlmostEqual(np.trapezoid(density, xs), 1, delta=0.05)

    def test_interval_handles_unsorted_observations_and_named_dimensions(self):
        samples = xr.DataArray(
            np.broadcast_to([3.0, 1.0, 2.0], (2, 100, 3)),
            dims=("chain", "draw", "observation"),
        ).transpose("observation", "draw", "chain")
        _, ax = plt.subplots()
        result = utils.plot_interval([3, 1, 2], samples, ax=ax)
        self.assertIs(result, ax)
        vertices = ax.collections[0].get_paths()[0].vertices
        self.assertTrue(np.isfinite(vertices).all())
        self.assertEqual(vertices[:, 0].min(), 1)
        self.assertEqual(vertices[:, 0].max(), 3)

    def test_forest_preserves_parameter_coordinates_and_subplot(self):
        posterior = xr.Dataset({
            "alpha": (("chain", "draw", "sex"), np.broadcast_to([1., 2.], (2, 100, 2)))
        }, coords={"sex": ["F", "M"]})
        trace = xr.DataTree.from_dict({"posterior": posterior})
        _, ax = plt.subplots()
        self.assertIs(utils.plot_forest(trace, ax=ax), ax)
        self.assertEqual([label.get_text() for label in ax.get_yticklabels()], ["alpha[F]", "alpha[M]"])
        np.testing.assert_allclose(ax.collections[-1].get_offsets()[:, 0], [1, 2])


if __name__ == "__main__":
    unittest.main()
