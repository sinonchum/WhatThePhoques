import unittest
import numpy as np
from region import guided, calibrate


class Fields(unittest.TestCase):
    def test_constant_field_stays_constant(self):
        rgb = np.random.default_rng(42).random((16, 16, 3), dtype=np.float32)
        np.testing.assert_allclose(
            guided(np.full((16, 16), 1.3, np.float32), rgb), 1.3, atol=1e-4
        )

    def test_identity_and_sign_equivariance(self):
        z = np.linspace(-2, 3, 256, dtype="f").reshape(16, 16)
        rgb = np.zeros((16, 16, 3), "f")
        np.testing.assert_array_equal(calibrate(z, rgb, (0, 0, 0)), z)
        np.testing.assert_allclose(
            calibrate(-z, rgb, (0.6, 0.8, 0.5)),
            -calibrate(z, rgb, (0.6, 0.8, 0.5)),
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
