import unittest

import torch

# Importing driss_torch registers the custom ops (keyed_random_uniform, etc.)
import driss_torch  # noqa: F401

# Assuming the op is registered under torch.ops.DrissTorch or accessible via the package
# Based on the registration: TORCH_LIBRARY(DrissTorch, m)
# It should be available as torch.ops.DrissTorch.keyed_random_uniform


class TestKeyedRandom(unittest.TestCase):
    def test_basic_generation(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        rows = 10
        cols = 100
        keys = torch.arange(rows, dtype=torch.int64, device="cuda")

        out = torch.ops.DrissTorch.keyed_random_uniform(keys, cols)

        self.assertEqual(out.shape, (rows, cols))
        self.assertEqual(out.device.type, "cuda")
        self.assertTrue((out >= 0).all())
        self.assertTrue((out < 1).all())

    def test_batch_invariance(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        rows = 32
        cols = 128

        # Create original keys
        keys = torch.randint(0, 100000, (rows,), dtype=torch.int64, device="cuda")

        # Generate reference output
        out_ref = torch.ops.DrissTorch.keyed_random_uniform(keys, cols)

        # Create a permutation
        perm = torch.randperm(rows, device="cuda")
        keys_perm = keys[perm]

        # Generate output with permuted keys
        out_perm = torch.ops.DrissTorch.keyed_random_uniform(keys_perm, cols)

        # The output corresponding to the permuted keys should match the permuted reference output
        # i.e. out_perm[i] should equal out_ref[perm[i]]
        # Or more simply: out_perm should be equal to out_ref[perm]

        out_ref_permuted = out_ref[perm]

        # Check equality
        # Since it's float, we should use close, but since it's deterministic integer-based math under the hood,
        # it should be exact if the implementation is correct.
        self.assertTrue(
            torch.equal(out_perm, out_ref_permuted),
            "Batch invariance failed! Permuting keys did not yield permuted rows.",
        )

    def test_determinism(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        rows = 5
        cols = 10
        keys = torch.randint(0, 1000, (rows,), dtype=torch.int64, device="cuda")

        out1 = torch.ops.DrissTorch.keyed_random_uniform(keys, cols)
        out2 = torch.ops.DrissTorch.keyed_random_uniform(keys, cols)

        self.assertTrue(
            torch.equal(out1, out2),
            "Determinism failed! Same keys produced different outputs.",
        )


if __name__ == "__main__":
    unittest.main()
