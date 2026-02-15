import copy
import unittest
from types import SimpleNamespace

import torch

from models.dual_tower.dual_tower import KVCacheBridge


class TestKVCacheBridge(unittest.TestCase):
    def _build_cfg(self, bridge_type: str, residual: bool, use_rmsnorm: bool = True):
        return SimpleNamespace(
            lm_hidden_dim=64,
            lm_n_heads=4,
            lm_n_blocks=3,
            kv_bridge_type=bridge_type,
            kv_bridge_mlp_ratio=2.0,
            kv_bridge_use_rmsnorm=use_rmsnorm,
            kv_bridge_residual=residual,
        )

    def _build_cache(self):
        return [
            {
                "key": torch.randn(2, 2, 7, 16),
                "value": torch.randn(2, 2, 7, 16),
            }
            for _ in range(3)
        ]

    def test_residual_linear_is_identity_initialized(self):
        cfg = self._build_cfg("linear", residual=True)
        bridge = KVCacheBridge(cfg)
        kv_cache = self._build_cache()
        original = copy.deepcopy(kv_cache)

        bridged = bridge(kv_cache)
        for old_layer, new_layer in zip(original, bridged):
            self.assertTrue(torch.allclose(old_layer["key"], new_layer["key"]))
            self.assertTrue(torch.allclose(old_layer["value"], new_layer["value"]))

    def test_residual_mlp_is_near_identity_initialized(self):
        cfg = self._build_cfg("mlp", residual=True)
        bridge = KVCacheBridge(cfg)
        kv_cache = self._build_cache()
        original = copy.deepcopy(kv_cache)

        bridged = bridge(kv_cache)
        for old_layer, new_layer in zip(original, bridged):
            key_delta = (new_layer["key"] - old_layer["key"]).abs().max().item()
            value_delta = (new_layer["value"] - old_layer["value"]).abs().max().item()
            self.assertLess(key_delta, 1e-2)
            self.assertLess(value_delta, 1e-2)

    def test_non_residual_linear_has_gradients(self):
        cfg = self._build_cfg("linear", residual=False, use_rmsnorm=False)
        bridge = KVCacheBridge(cfg)
        kv_cache = self._build_cache()

        for layer in kv_cache:
            layer["key"].requires_grad_(True)
            layer["value"].requires_grad_(True)

        bridged = bridge(kv_cache)
        loss = sum(layer["key"].sum() + layer["value"].sum() for layer in bridged)
        loss.backward()

        grads = [p.grad for p in bridge.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads))


if __name__ == "__main__":
    unittest.main()
