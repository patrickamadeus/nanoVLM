import unittest
from unittest.mock import patch

import torch

from models.config import VLMConfig
from models.dual_tower.dual_tower import DualTowerVLM
from models.language_model import LanguageModel, LanguageModelGroupedQueryAttention


class _FakeTokenizer:
    def __init__(self, token_to_id: dict[str, int]):
        self._token_to_id = token_to_id
        self.image_token_id = token_to_id["<|image|>"]
        # Disable early-stop in generation so decode-step counts are stable in tests.
        self.eos_token_id = None

    def convert_tokens_to_ids(self, token: str):
        return self._token_to_id.get(token, -1)


class TestDualTowerRoPE(unittest.TestCase):
    def _build_dual_cfg(self, left_mask_scope: str, right_prefill_mode: str = "full") -> VLMConfig:
        return VLMConfig(
            vit_model_type="testing",
            vit_hidden_dim=32,
            vit_inter_dim=64,
            vit_patch_size=8,
            vit_img_size=16,
            vit_n_heads=4,
            vit_n_blocks=1,
            vit_dropout=0.0,
            lm_model_type="testing",
            lm_tokenizer="testing",
            lm_hidden_dim=32,
            lm_inter_dim=64,
            lm_rms_eps=1e-5,
            lm_re_base=10000.0,
            lm_max_position_embeddings=128,
            lm_attn_scaling=1.0,
            lm_vocab_size=256,
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_dropout=0.0,
            lm_n_blocks=2,
            lm_use_tokens=False,
            lm_tie_weights=True,
            mp_pixel_shuffle_factor=2,
            mp_image_token_length=4,
            vlm_extra_tokens={
                "image_token": "<|image|>",
                "global_image_token": "<|global_image|>",
                "r1c1": "<row_1_col_1>",
            },
            left_mask_scope=left_mask_scope,
            right_prefill_mode=right_prefill_mode,
            use_kv_bridge=False,
        )

    def _build_language_cfg(self):
        return type(
            "Cfg",
            (),
            {
                "lm_hidden_dim": 64,
                "lm_inter_dim": 128,
                "lm_rms_eps": 1e-5,
                "lm_re_base": 10000.0,
                "lm_max_position_embeddings": 256,
                "lm_attn_scaling": 1.0,
                "lm_pad_aware_rope": False,
                "lm_vocab_size": 128,
                "lm_n_heads": 4,
                "lm_n_kv_heads": 2,
                "lm_dropout": 0.0,
                "lm_n_blocks": 2,
                "lm_use_tokens": True,
                "lm_tie_weights": True,
            },
        )()

    def _build_tokenizer(self):
        return _FakeTokenizer(
            {
                "<|image|>": 101,
                "<|global_image|>": 102,
                "<row_1_col_1>": 103,
            }
        )

    def test_dual_prefill_attention_mask_matches_legacy_prefix_behavior(self):
        replace_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        key_valid_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
        got = LanguageModelGroupedQueryAttention._build_dual_prefill_attn_mask(
            replace_mask=replace_mask,
            key_valid_mask=key_valid_mask,
        )
        expected = torch.tensor(
            [
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1],
                ]
            ],
            dtype=torch.bool,
        )
        self.assertTrue(torch.equal(got, expected))

    def test_pad_aware_rope_compacts_prefill_positions_when_enabled(self):
        cfg = self._build_language_cfg()
        cfg.lm_pad_aware_rope = True
        model = LanguageModel(cfg).eval()

        position_ids = torch.arange(0, 6).unsqueeze(0)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1]], dtype=torch.long)

        cos_pad_aware, sin_pad_aware = model.rotary_embd(position_ids, attention_mask=attention_mask)
        compact_positions = torch.tensor([[0, 0, 0, 1, 2, 3]], dtype=torch.long)
        cos_compact, sin_compact = model.rotary_embd(compact_positions, attention_mask=None)

        # Pad-aware mode zeros pad locations and matches compact positions on valid tokens.
        valid = attention_mask.bool()
        self.assertTrue(torch.allclose(cos_pad_aware[:, valid[0], :], cos_compact[:, valid[0], :], atol=1e-6))
        self.assertTrue(torch.allclose(sin_pad_aware[:, valid[0], :], sin_compact[:, valid[0], :], atol=1e-6))
        self.assertTrue(torch.equal(cos_pad_aware[:, ~valid[0], :], torch.zeros_like(cos_pad_aware[:, ~valid[0], :])))
        self.assertTrue(torch.equal(sin_pad_aware[:, ~valid[0], :], torch.zeros_like(sin_pad_aware[:, ~valid[0], :])))

    def test_left_mask_scope_masks_match_expected_regions(self):
        fake_tokenizer = self._build_tokenizer()
        input_ids = torch.tensor([[5, 6, 101, 7, 102, 8, 0, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.long)

        expected = {
            "visual_only": torch.tensor([[0, 0, 1, 0, 1, 0, 0, 0]], dtype=torch.bool),
            "visual_sys": torch.tensor([[1, 1, 1, 0, 1, 0, 0, 0]], dtype=torch.bool),
            "full": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool),
        }

        for scope, expected_mask in expected.items():
            with self.subTest(scope=scope):
                cfg = self._build_dual_cfg(scope)
                with patch("models.vision_language_model.get_tokenizer", return_value=fake_tokenizer):
                    model = DualTowerVLM(cfg, load_backbone=False)
                got = model._build_scope_mask(input_ids, attention_mask)
                self.assertTrue(torch.equal(got, expected_mask))

    def test_generate_decode_start_pos_continues_from_kv_length_for_all_scopes(self):
        torch.manual_seed(0)
        fake_tokenizer = self._build_tokenizer()
        prompt = torch.tensor([[11, 12, 101, 13, 14, 15]], dtype=torch.long)
        # Trailing pads intentionally included; decode continuity must still follow KV length.
        prompt_attention = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
        max_new_tokens = 3

        for scope in ("visual_only", "visual_sys", "full"):
            with self.subTest(scope=scope):
                cfg = self._build_dual_cfg(scope)
                with patch("models.vision_language_model.get_tokenizer", return_value=fake_tokenizer):
                    model = DualTowerVLM(cfg, load_backbone=False)

                recorded = []
                original_forward = model.right_tower.forward

                def wrapped_forward(*, x, attention_mask=None, kv_cache=None, start_pos=0):
                    if isinstance(start_pos, torch.Tensor):
                        start_val = int(start_pos.reshape(-1)[0].item())
                    else:
                        start_val = int(start_pos)
                    kv_len = None
                    if kv_cache is not None and kv_cache[0] is not None and kv_cache[0].get("key") is not None:
                        kv_len = int(kv_cache[0]["key"].size(2))
                    recorded.append((start_val, kv_len, int(x.size(1))))
                    return original_forward(
                        x=x,
                        attention_mask=attention_mask,
                        kv_cache=kv_cache,
                        start_pos=start_pos,
                    )

                model.right_tower.forward = wrapped_forward
                _ = model.generate(
                    input_ids=prompt,
                    images=None,
                    attention_mask=prompt_attention,
                    max_new_tokens=max_new_tokens,
                    greedy=True,
                )

                # First call is prefill.
                self.assertGreaterEqual(len(recorded), 1 + max_new_tokens)
                prefill_start, _, prefill_t = recorded[0]
                self.assertEqual(prefill_start, 0)
                self.assertEqual(prefill_t, prompt.size(1))

                # Decode calls must use start_pos == cached KV length, then keep incrementing.
                decode_calls = recorded[1 : 1 + max_new_tokens]
                expected_first_decode_start = prompt.size(1)
                self.assertEqual(decode_calls[0][0], expected_first_decode_start)
                self.assertNotEqual(decode_calls[0][0], 0)

                expected_starts = [prompt.size(1) + i for i in range(max_new_tokens)]
                got_starts = [item[0] for item in decode_calls]
                self.assertEqual(got_starts, expected_starts)

                for start_pos, kv_len, t_curr in decode_calls:
                    self.assertEqual(t_curr, 1)
                    self.assertEqual(start_pos, kv_len)

    def test_generate_non_donor_only_uses_single_query_prefill(self):
        torch.manual_seed(0)
        fake_tokenizer = self._build_tokenizer()
        prompt = torch.tensor([[11, 12, 101, 13, 14, 15]], dtype=torch.long)
        prompt_attention = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long)

        cfg = self._build_dual_cfg("full", right_prefill_mode="non_donor_only")
        with patch("models.vision_language_model.get_tokenizer", return_value=fake_tokenizer):
            model = DualTowerVLM(cfg, load_backbone=False)

        recorded = []
        original_forward = model.right_tower.forward

        def wrapped_forward(*, x, attention_mask=None, kv_cache=None, start_pos=0):
            if isinstance(start_pos, torch.Tensor):
                start_val = int(start_pos.reshape(-1)[0].item())
            else:
                start_val = int(start_pos)
            kv_len = None
            query_only = False
            if kv_cache is not None and kv_cache[0] is not None:
                query_only = bool(kv_cache[0].get("query_only", False))
                if kv_cache[0].get("key") is not None:
                    kv_len = int(kv_cache[0]["key"].size(2))
            recorded.append((start_val, kv_len, int(x.size(1)), query_only))
            return original_forward(
                x=x,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )

        model.right_tower.forward = wrapped_forward
        _ = model.generate(
            input_ids=prompt,
            images=None,
            attention_mask=prompt_attention,
            max_new_tokens=2,
            greedy=True,
        )

        # First call is bootstrap: one query token attending over full cached prompt.
        self.assertGreaterEqual(len(recorded), 3)
        first_start, first_kv_len, first_t, first_query_only = recorded[0]
        self.assertEqual(first_t, 1)
        self.assertEqual(first_kv_len, prompt.size(1))
        self.assertEqual(first_start, prompt.size(1) - 1)
        self.assertTrue(first_query_only)

        # First decode step starts exactly at cached length, using standard append mode.
        dec_start, dec_kv_len, dec_t, dec_query_only = recorded[1]
        self.assertEqual(dec_t, 1)
        self.assertEqual(dec_start, prompt.size(1))
        self.assertEqual(dec_start, dec_kv_len)
        self.assertFalse(dec_query_only)

    def test_generate_visual_sys_bootstrap_uses_unprocessed_query_span(self):
        torch.manual_seed(0)
        fake_tokenizer = self._build_tokenizer()
        # donor mask for visual_sys => indices [0,1,2,4], unprocessed query span => [3,5]
        prompt = torch.tensor([[11, 12, 101, 13, 102, 15]], dtype=torch.long)
        prompt_attention = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long)

        cfg = self._build_dual_cfg("visual_sys", right_prefill_mode="non_donor_only")
        with patch("models.vision_language_model.get_tokenizer", return_value=fake_tokenizer):
            model = DualTowerVLM(cfg, load_backbone=False)

        recorded = []
        original_forward = model.right_tower.forward

        def wrapped_forward(*, x, attention_mask=None, kv_cache=None, start_pos=0):
            kv_len = None
            query_only = False
            if kv_cache is not None and kv_cache[0] is not None:
                query_only = bool(kv_cache[0].get("query_only", False))
                if kv_cache[0].get("key") is not None:
                    kv_len = int(kv_cache[0]["key"].size(2))
            if isinstance(start_pos, torch.Tensor):
                start_repr = tuple(start_pos.shape)
                start_vals = start_pos.detach().cpu().tolist()
            else:
                start_repr = "scalar"
                start_vals = int(start_pos)
            recorded.append((start_repr, start_vals, int(x.size(1)), kv_len, query_only))
            return original_forward(
                x=x,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                start_pos=start_pos,
            )

        model.right_tower.forward = wrapped_forward
        _ = model.generate(
            input_ids=prompt,
            images=None,
            attention_mask=prompt_attention,
            max_new_tokens=1,
            greedy=True,
        )

        self.assertGreaterEqual(len(recorded), 2)
        first_shape, first_start, first_t, first_kv_len, first_query_only = recorded[0]
        self.assertEqual(first_shape, (1, 2))
        self.assertEqual(first_start, [[3, 5]])
        self.assertEqual(first_t, 2)
        self.assertEqual(first_kv_len, prompt.size(1))
        self.assertTrue(first_query_only)

    def test_language_model_decode_matches_full_pass_when_rope_position_continues(self):
        torch.manual_seed(1)
        cfg = self._build_language_cfg()
        model = LanguageModel(cfg).eval()

        batch_size = 2
        prompt_len = 6
        prompt_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, prompt_len))
        prompt_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 0],
            ],
            dtype=torch.long,
        )
        next_token = torch.randint(0, cfg.lm_vocab_size, (batch_size, 1))
        full_ids = torch.cat([prompt_ids, next_token], dim=1)
        full_mask = torch.cat([prompt_mask, torch.ones((batch_size, 1), dtype=torch.long)], dim=1)

        # Reference: single full forward.
        full_logits, _ = model(full_ids, attention_mask=full_mask, start_pos=0)
        ref_last_logits = full_logits[:, -1, :]

        # Cached decode with continued position (what full-scope dualtower decode requires).
        _, kv_cache = model(prompt_ids, attention_mask=prompt_mask, start_pos=0)
        decode_logits_ok, _ = model(
            next_token,
            attention_mask=full_mask,
            kv_cache=kv_cache,
            start_pos=prompt_len,
        )
        got_last_logits = decode_logits_ok[:, 0, :]
        self.assertTrue(torch.allclose(ref_last_logits, got_last_logits, atol=1e-5))

        # Negative check: restarting RoPE at 0 should diverge from reference.
        _, kv_cache_bad = model(prompt_ids, attention_mask=prompt_mask, start_pos=0)
        decode_logits_bad, _ = model(
            next_token,
            attention_mask=full_mask,
            kv_cache=kv_cache_bad,
            start_pos=0,
        )
        bad_last_logits = decode_logits_bad[:, 0, :]
        self.assertFalse(torch.allclose(ref_last_logits, bad_last_logits, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
