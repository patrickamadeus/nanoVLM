import copy
import unittest

import torch

from models.dual_tower.dual_tower import RightKVCacheGates


class TestRightKVCacheGates(unittest.TestCase):
    @staticmethod
    def _build_cache(layer_values: list[float]):
        cache = []
        for val in layer_values:
            tensor = torch.full((1, 1, 2, 1), float(val))
            cache.append({"key": tensor.clone(), "value": tensor.clone()})
        return cache

    def test_unnormalized_triangular_mix(self):
        gates = RightKVCacheGates(3, normalize=False, init_offdiag=0.5)
        kv_cache = self._build_cache([1.0, 2.0, 3.0])
        mixed = gates(copy.deepcopy(kv_cache))

        self.assertTrue(torch.allclose(mixed[0]["key"], torch.full((1, 1, 2, 1), 3.5)))
        self.assertTrue(torch.allclose(mixed[1]["key"], torch.full((1, 1, 2, 1), 3.5)))
        self.assertTrue(torch.allclose(mixed[2]["key"], torch.full((1, 1, 2, 1), 3.0)))

        self.assertTrue(torch.allclose(mixed[0]["value"], torch.full((1, 1, 2, 1), 3.5)))
        self.assertTrue(torch.allclose(mixed[1]["value"], torch.full((1, 1, 2, 1), 3.5)))
        self.assertTrue(torch.allclose(mixed[2]["value"], torch.full((1, 1, 2, 1), 3.0)))

    def test_normalized_triangular_mix(self):
        gates = RightKVCacheGates(3, normalize=True, init_offdiag=0.5)
        kv_cache = self._build_cache([1.0, 2.0, 3.0])
        mixed = gates(copy.deepcopy(kv_cache))

        self.assertTrue(torch.allclose(mixed[0]["key"], torch.full((1, 1, 2, 1), 1.75)))
        self.assertTrue(
            torch.allclose(
                mixed[1]["key"],
                torch.full((1, 1, 2, 1), 2.3333333),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(mixed[2]["key"], torch.full((1, 1, 2, 1), 3.0)))

        self.assertTrue(torch.allclose(mixed[0]["value"], torch.full((1, 1, 2, 1), 1.75)))
        self.assertTrue(
            torch.allclose(
                mixed[1]["value"],
                torch.full((1, 1, 2, 1), 2.3333333),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(mixed[2]["value"], torch.full((1, 1, 2, 1), 3.0)))

    def test_only_last_token_mix_preserves_prefix(self):
        gates = RightKVCacheGates(3, normalize=False, init_offdiag=0.5)
        kv_cache = []
        for val in [1.0, 2.0, 3.0]:
            tensor = torch.full((1, 1, 3, 1), float(val))
            tensor[:, :, 0, :] = -1.0
            tensor[:, :, 1, :] = -2.0
            kv_cache.append({"key": tensor.clone(), "value": tensor.clone()})

        mixed = gates(copy.deepcopy(kv_cache), only_last_token=True)

        for layer in mixed:
            self.assertTrue(torch.allclose(layer["key"][:, :, 0, :], torch.full((1, 1, 1), -1.0)))
            self.assertTrue(torch.allclose(layer["key"][:, :, 1, :], torch.full((1, 1, 1), -2.0)))
            self.assertTrue(torch.allclose(layer["value"][:, :, 0, :], torch.full((1, 1, 1), -1.0)))
            self.assertTrue(torch.allclose(layer["value"][:, :, 1, :], torch.full((1, 1, 1), -2.0)))

        self.assertTrue(torch.allclose(mixed[0]["key"][:, :, 2, :], torch.full((1, 1, 1), 3.5)))
        self.assertTrue(torch.allclose(mixed[1]["key"][:, :, 2, :], torch.full((1, 1, 1), 3.5)))
        self.assertTrue(torch.allclose(mixed[2]["key"][:, :, 2, :], torch.full((1, 1, 1), 3.0)))
        self.assertTrue(torch.allclose(mixed[0]["value"][:, :, 2, :], torch.full((1, 1, 1), 3.5)))
        self.assertTrue(torch.allclose(mixed[1]["value"][:, :, 2, :], torch.full((1, 1, 1), 3.5)))
        self.assertTrue(torch.allclose(mixed[2]["value"][:, :, 2, :], torch.full((1, 1, 1), 3.0)))


if __name__ == "__main__":
    unittest.main()
