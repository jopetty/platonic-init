from __future__ import annotations

import unittest

import torch

from platonic_init.rebasin import align_states_for_pca


class RebasinTests(unittest.TestCase):
    def test_align_states_recovers_mlp_permutation(self) -> None:
        hidden = 4
        mlp = 8
        gen = torch.Generator().manual_seed(123)
        base = {
            "transformer.h.0.mlp.c_fc.weight": torch.randn(hidden, mlp, generator=gen),
            "transformer.h.0.mlp.c_fc.bias": torch.randn(mlp, generator=gen),
            "transformer.h.0.mlp.c_proj.weight": torch.randn(
                mlp, hidden, generator=gen
            ),
            "transformer.h.0.mlp.c_proj.bias": torch.randn(hidden, generator=gen),
            "transformer.h.0.ln_1.weight": torch.randn(hidden, generator=gen),
        }

        perm = torch.tensor([3, 1, 6, 2, 4, 5, 0, 7], dtype=torch.long)
        permuted = {
            "transformer.h.0.mlp.c_fc.weight": base["transformer.h.0.mlp.c_fc.weight"][
                :, perm
            ],
            "transformer.h.0.mlp.c_fc.bias": base["transformer.h.0.mlp.c_fc.bias"][
                perm
            ],
            "transformer.h.0.mlp.c_proj.weight": base[
                "transformer.h.0.mlp.c_proj.weight"
            ][perm, :],
            "transformer.h.0.mlp.c_proj.bias": base[
                "transformer.h.0.mlp.c_proj.bias"
            ].clone(),
            "transformer.h.0.ln_1.weight": base["transformer.h.0.ln_1.weight"].clone(),
        }

        aligned, report = align_states_for_pca([base, permuted], max_iter=30, seed=0)
        self.assertEqual(report["num_permutations"], 1)
        self.assertTrue(
            torch.allclose(
                aligned[0]["transformer.h.0.mlp.c_fc.weight"],
                aligned[1]["transformer.h.0.mlp.c_fc.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                aligned[0]["transformer.h.0.mlp.c_fc.bias"],
                aligned[1]["transformer.h.0.mlp.c_fc.bias"],
            )
        )
        self.assertTrue(
            torch.allclose(
                aligned[0]["transformer.h.0.mlp.c_proj.weight"],
                aligned[1]["transformer.h.0.mlp.c_proj.weight"],
            )
        )

    def test_align_states_recovers_attention_head_permutation(self) -> None:
        hidden = 6
        num_heads = 3
        head_dim = hidden // num_heads
        gen = torch.Generator().manual_seed(321)
        base = {
            "transformer.h.0.attn.c_attn.weight": torch.randn(
                hidden, 3 * hidden, generator=gen
            ),
            "transformer.h.0.attn.c_attn.bias": torch.randn(3 * hidden, generator=gen),
            "transformer.h.0.attn.c_proj.weight": torch.randn(
                hidden, hidden, generator=gen
            ),
        }

        perm = torch.tensor([2, 0, 1], dtype=torch.long)
        qkv_w = base["transformer.h.0.attn.c_attn.weight"].reshape(
            hidden, 3, num_heads, head_dim
        )
        qkv_b = base["transformer.h.0.attn.c_attn.bias"].reshape(3, num_heads, head_dim)
        proj_w = base["transformer.h.0.attn.c_proj.weight"].reshape(
            num_heads, head_dim, hidden
        )
        permuted = {
            "transformer.h.0.attn.c_attn.weight": qkv_w[:, :, perm, :].reshape(
                hidden, 3 * hidden
            ),
            "transformer.h.0.attn.c_attn.bias": qkv_b[:, perm, :].reshape(3 * hidden),
            "transformer.h.0.attn.c_proj.weight": proj_w[perm, :, :].reshape(
                hidden, hidden
            ),
        }

        aligned, report = align_states_for_pca(
            [base, permuted], max_iter=30, seed=0, num_attention_heads=num_heads
        )
        self.assertEqual(report["num_permutations"], 1)
        self.assertTrue(
            torch.allclose(
                aligned[0]["transformer.h.0.attn.c_attn.weight"],
                aligned[1]["transformer.h.0.attn.c_attn.weight"],
            )
        )
        self.assertTrue(
            torch.allclose(
                aligned[0]["transformer.h.0.attn.c_attn.bias"],
                aligned[1]["transformer.h.0.attn.c_attn.bias"],
            )
        )
        self.assertTrue(
            torch.allclose(
                aligned[0]["transformer.h.0.attn.c_proj.weight"],
                aligned[1]["transformer.h.0.attn.c_proj.weight"],
            )
        )


if __name__ == "__main__":
    unittest.main()
