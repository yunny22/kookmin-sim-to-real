import unittest

import numpy as np

from xycar_rl.sampling import signed_uniform


class SamplingTest(unittest.TestCase):
    def test_signed_uniform_respects_nonzero_magnitude(self):
        rng = np.random.default_rng(4)
        values = [signed_uniform(rng, 0.08, 0.22) for _ in range(200)]
        self.assertTrue(all(0.08 <= abs(value) <= 0.22 for value in values))
        self.assertTrue(any(value < 0.0 for value in values))
        self.assertTrue(any(value > 0.0 for value in values))


if __name__ == "__main__":
    unittest.main()
