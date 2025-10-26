"""Tests for observation encoders.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import torch

from mip.encoders import (
    IdentityEncoder,
    MultiImageObsEncoder,
)


class TestIdentityEncoder:
    """Test suite for the IdentityEncoder class."""

    def test_identity_encoder_creation(self):
        """Test creating an IdentityEncoder."""
        encoder = IdentityEncoder(dropout=0.25)
        assert encoder is not None

    def test_identity_encoder_forward_with_mask(self):
        """Test IdentityEncoder forward pass with mask."""
        encoder = IdentityEncoder(dropout=0.0)
        encoder.eval()

        bs = 4
        obs_dim = 10
        condition = torch.randn(bs, obs_dim)
        mask = torch.ones(bs)

        with torch.no_grad():
            output = encoder(condition, mask)

        assert output.shape == condition.shape
        assert torch.allclose(output, condition)


class TestMultiImageObsEncoder:
    """Test suite for the MultiImageObsEncoder class."""

    def test_multi_image_encoder_creation_rgb_only(self):
        """Test creating a MultiImageObsEncoder with RGB input only."""
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
        )
        assert encoder is not None

    def test_multi_image_encoder_creation_mixed_inputs(self):
        """Test creating a MultiImageObsEncoder with mixed RGB and low-dim inputs."""
        shape_meta = {
            "obs": {
                "rgb": {"shape": [3, 224, 224], "type": "rgb"},
                "state": {"shape": [10], "type": "low_dim"},
            }
        }
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
        )
        assert encoder is not None
