import copy
import unittest
from types import SimpleNamespace

import torch

from models.language_model import LanguageModel


class TestDualTowerAttentionGating(unittest.TestCase):
    def _build_cfg(
        self,
        *,
        gate_mode: str,
        gate_scope: str = "all",
        gate_min: float | None = 0.0,
        gate_max: float | None = 1.0,
    ):
        return SimpleNamespace(
            lm_hidden_dim=64,
            lm_inter_dim=128,
            lm_rms_eps=1e-5,
            lm_re_base=10000.0,
            lm_max_position_embeddings=256,
            lm_attn_scaling=1.0,
            lm_vocab_size=257,
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_dropout=0.0,
            lm_n_blocks=1,
            lm_use_tokens=True,
            lm_tie_weights=False,
            right_attn_gate_mode=gate_mode,
            right_attn_gate_granularity="elementwise",
            right_attn_gate_scope=gate_scope,
            right_attn_gate_min=gate_min,
            right_attn_gate_max=gate_max,
        )

    @staticmethod
    def _copy_shared_parameters(src: LanguageModel, dst: LanguageModel) -> None:
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        for key, value in src_state.items():
            if key in dst_state and dst_state[key].shape == value.shape:
                dst_state[key] = value.clone()
        dst.load_state_dict(dst_state, strict=False)

    def test_gating_modes_preserve_cache_shapes(self):
        torch.manual_seed(0)
        batch_size = 2
        seq_len = 6

        for gate_mode in ("none", "kv", "o_proj", "kv+o_proj"):
            cfg = self._build_cfg(gate_mode=gate_mode)
            model = LanguageModel(cfg).eval()
            input_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len))

            with torch.no_grad():
                prefill_output, kv_cache = model(input_ids, start_pos=0)
                decode_input = torch.randint(0, cfg.lm_vocab_size, (batch_size, 1))
                decode_output, kv_cache = model(decode_input, kv_cache=kv_cache, start_pos=seq_len)

            self.assertEqual(tuple(prefill_output.shape), (batch_size, seq_len, cfg.lm_vocab_size))
            self.assertEqual(tuple(decode_output.shape), (batch_size, 1, cfg.lm_vocab_size))
            self.assertEqual(len(kv_cache), cfg.lm_n_blocks)

            layer_cache = kv_cache[0]
            self.assertEqual(tuple(layer_cache["key"].shape), (batch_size, cfg.lm_n_kv_heads, seq_len + 1, 16))
            self.assertEqual(tuple(layer_cache["value"].shape), (batch_size, cfg.lm_n_kv_heads, seq_len + 1, 16))
            self.assertEqual(tuple(layer_cache["donor_key_mask"].shape), (batch_size, seq_len + 1))
            self.assertFalse(layer_cache["donor_key_mask"][:, -1].any().item())
            if gate_mode in {"kv", "kv+o_proj"}:
                self.assertIn("k_gate", layer_cache)
                self.assertIn("v_gate", layer_cache)
                self.assertEqual(tuple(layer_cache["k_gate"].shape), (batch_size, cfg.lm_n_kv_heads, seq_len + 1, 16))
                self.assertEqual(tuple(layer_cache["v_gate"].shape), (batch_size, cfg.lm_n_kv_heads, seq_len + 1, 16))
            else:
                self.assertNotIn("k_gate", layer_cache)
                self.assertNotIn("v_gate", layer_cache)

    def test_projection_splits_follow_gate_mode(self):
        base_v_out = 16 * 2  # head_dim * n_kv_heads
        base_k_out = base_v_out
        cases = [
            ("none", 64, base_k_out, base_v_out),
            ("o_proj", 128, base_k_out, base_v_out),
            ("kv", 64, base_k_out * 2, base_v_out * 2),
            ("kv+o_proj", 128, base_k_out * 2, base_v_out * 2),
        ]
        for gate_mode, expected_q_out, expected_k_out, expected_v_out in cases:
            cfg = self._build_cfg(gate_mode=gate_mode)
            model = LanguageModel(cfg).eval()
            attn = model.blocks[0].attn
            self.assertEqual(attn.q_proj.out_features, expected_q_out)
            self.assertEqual(attn.k_proj.out_features, expected_k_out)
            self.assertEqual(attn.v_proj.out_features, expected_v_out)

    def test_donor_only_scope_is_noop_without_donor_tokens(self):
        torch.manual_seed(2)
        base_cfg = self._build_cfg(gate_mode="none")
        gated_cfg = self._build_cfg(
            gate_mode="kv",
            gate_scope="donor_only",
            gate_min=0.0,
            gate_max=0.0,
        )

        baseline = LanguageModel(base_cfg).eval()
        gated = LanguageModel(gated_cfg).eval()
        self._copy_shared_parameters(baseline, gated)

        batch_size = 1
        seq_len = 6
        input_ids = torch.randint(0, base_cfg.lm_vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        donor_kv = {
            "key": torch.randn(batch_size, base_cfg.lm_n_kv_heads, seq_len, 16),
            "value": torch.randn(batch_size, base_cfg.lm_n_kv_heads, seq_len, 16),
            "needs_dual_prefill": True,
            "replace_mask": torch.zeros((batch_size, seq_len), dtype=torch.bool),
        }

        with torch.no_grad():
            base_output, _ = baseline(
                input_ids,
                attention_mask=attention_mask,
                kv_cache=[copy.deepcopy(donor_kv)],
                start_pos=0,
            )
            gated_output, _ = gated(
                input_ids,
                attention_mask=attention_mask,
                kv_cache=[copy.deepcopy(donor_kv)],
                start_pos=0,
            )

        self.assertTrue(torch.allclose(base_output, gated_output, atol=1e-5, rtol=1e-5))

    def test_donor_only_scope_changes_output_when_donor_tokens_present(self):
        torch.manual_seed(3)
        base_cfg = self._build_cfg(gate_mode="none")
        gated_cfg = self._build_cfg(
            gate_mode="kv",
            gate_scope="donor_only",
            gate_min=0.0,
            gate_max=0.0,
        )

        baseline = LanguageModel(base_cfg).eval()
        gated = LanguageModel(gated_cfg).eval()
        self._copy_shared_parameters(baseline, gated)

        batch_size = 1
        seq_len = 6
        input_ids = torch.randint(0, base_cfg.lm_vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        donor_kv = {
            "key": torch.randn(batch_size, base_cfg.lm_n_kv_heads, seq_len, 16),
            "value": torch.randn(batch_size, base_cfg.lm_n_kv_heads, seq_len, 16),
            "needs_dual_prefill": True,
            "replace_mask": torch.ones((batch_size, seq_len), dtype=torch.bool),
        }

        with torch.no_grad():
            base_output, _ = baseline(
                input_ids,
                attention_mask=attention_mask,
                kv_cache=[copy.deepcopy(donor_kv)],
                start_pos=0,
            )
            gated_output, _ = gated(
                input_ids,
                attention_mask=attention_mask,
                kv_cache=[copy.deepcopy(donor_kv)],
                start_pos=0,
            )

        max_delta = (base_output - gated_output).abs().max().item()
        self.assertGreater(max_delta, 1e-4)

    def test_o_proj_mode_builds(self):
        cfg = self._build_cfg(gate_mode="o_proj")
        model = LanguageModel(cfg).eval()
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
