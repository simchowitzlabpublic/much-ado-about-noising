"""Tests for observation encoders.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import torch
import torch.nn as nn

from mip.encoders import (
    CropRandomizer,
    IdentityEncoder,
    MultiImageObsEncoder,
    crop_image_from_indices,
    get_resnet,
    sample_random_image_crops,
)
from mip.networks.mlp import MLP, VanillaMLP


class TestIdentityEncoder:
    """Test suite for the IdentityEncoder class."""

    def test_identity_encoder_creation(self):
        """Test creating an IdentityEncoder."""
        encoder = IdentityEncoder(dropout=0.25)
        assert encoder is not None

    def test_identity_encoder_forward_no_mask(self):
        """Test IdentityEncoder forward pass without mask."""
        encoder = IdentityEncoder(dropout=0.0)
        encoder.eval()

        bs = 4
        obs_dim = 10
        condition = torch.randn(bs, obs_dim)

        with torch.no_grad():
            output = encoder(condition)

        assert output.shape == condition.shape
        assert torch.allclose(output, condition)

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

    def test_identity_encoder_dropout_training(self):
        """Test that IdentityEncoder applies dropout during training."""
        encoder = IdentityEncoder(dropout=1.0)  # 100% dropout
        encoder.train()

        bs = 4
        obs_dim = 10
        condition = torch.randn(bs, obs_dim)

        output = encoder(condition)

        # With 100% dropout, output should be all zeros
        assert output.shape == condition.shape
        # In training mode with high dropout, most values should be masked
        assert torch.abs(output).sum() < torch.abs(condition).sum()

    def test_identity_encoder_2d_input(self):
        """Test IdentityEncoder with 2D input."""
        encoder = IdentityEncoder(dropout=0.0)
        encoder.eval()

        bs = 4
        seq_len = 8
        obs_dim = 10
        condition = torch.randn(bs, seq_len, obs_dim)

        with torch.no_grad():
            output = encoder(condition)

        assert output.shape == condition.shape


class TestCropRandomizer:
    """Test suite for the CropRandomizer class."""

    def test_crop_randomizer_creation(self):
        """Test creating a CropRandomizer."""
        randomizer = CropRandomizer(
            input_shape=(3, 240, 240),
            crop_height=216,
            crop_width=216,
            num_crops=1,
            pos_enc=False,
        )
        assert randomizer is not None

    def test_crop_randomizer_output_shape_in(self):
        """Test CropRandomizer output_shape_in method."""
        randomizer = CropRandomizer(
            input_shape=(3, 240, 240),
            crop_height=216,
            crop_width=216,
            num_crops=1,
            pos_enc=False,
        )
        output_shape = randomizer.output_shape_in()
        assert output_shape == [3, 216, 216]

    def test_crop_randomizer_output_shape_in_with_pos_enc(self):
        """Test CropRandomizer output_shape_in with position encoding."""
        randomizer = CropRandomizer(
            input_shape=(3, 240, 240),
            crop_height=216,
            crop_width=216,
            num_crops=1,
            pos_enc=True,
        )
        output_shape = randomizer.output_shape_in()
        assert output_shape == [5, 216, 216]  # 3 + 2 for position encoding

    def test_crop_randomizer_forward_in_eval(self):
        """Test CropRandomizer forward_in during evaluation (center crop)."""
        randomizer = CropRandomizer(
            input_shape=(3, 240, 240),
            crop_height=216,
            crop_width=216,
            num_crops=1,
            pos_enc=False,
        )
        randomizer.eval()

        bs = 4
        images = torch.randn(bs, 3, 240, 240)

        with torch.no_grad():
            output = randomizer.forward_in(images)

        assert output.shape == (bs, 3, 216, 216)

    def test_crop_randomizer_forward_in_train(self):
        """Test CropRandomizer forward_in during training (random crop)."""
        randomizer = CropRandomizer(
            input_shape=(3, 240, 240),
            crop_height=216,
            crop_width=216,
            num_crops=1,
            pos_enc=False,
        )
        randomizer.train()

        bs = 4
        images = torch.randn(bs, 3, 240, 240)

        output = randomizer.forward_in(images)

        assert output.shape == (bs, 3, 216, 216)

    def test_crop_randomizer_forward_out(self):
        """Test CropRandomizer forward_out."""
        randomizer = CropRandomizer(
            input_shape=(3, 240, 240),
            crop_height=216,
            crop_width=216,
            num_crops=4,
            pos_enc=False,
        )

        bs = 8
        num_crops = 4
        features = torch.randn(bs * num_crops, 512)

        output = randomizer.forward_out(features)

        assert output.shape == (bs, 512)

    def test_crop_randomizer_multiple_crops(self):
        """Test CropRandomizer with multiple crops."""
        num_crops = 4
        randomizer = CropRandomizer(
            input_shape=(3, 240, 240),
            crop_height=216,
            crop_width=216,
            num_crops=num_crops,
            pos_enc=False,
        )
        randomizer.eval()

        bs = 2
        images = torch.randn(bs, 3, 240, 240)

        with torch.no_grad():
            output = randomizer.forward_in(images)

        assert output.shape == (bs * num_crops, 3, 216, 216)


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

    def test_multi_image_encoder_forward_rgb_only(self):
        """Test MultiImageObsEncoder forward pass with RGB input only."""
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
            use_seq=False,
        )
        encoder.eval()

        bs = 4
        obs_dict = {"rgb": torch.randn(bs, 3, 224, 224)}

        with torch.no_grad():
            output = encoder(obs_dict)

        assert output.shape == (bs, 256)

    def test_multi_image_encoder_forward_mixed_inputs(self):
        """Test MultiImageObsEncoder forward pass with mixed inputs."""
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
            use_seq=False,
        )
        encoder.eval()

        bs = 4
        obs_dict = {
            "rgb": torch.randn(bs, 3, 224, 224),
            "state": torch.randn(bs, 10),
        }

        with torch.no_grad():
            output = encoder(obs_dict)

        assert output.shape == (bs, 256)

    def test_multi_image_encoder_with_crop(self):
        """Test MultiImageObsEncoder with cropping."""
        shape_meta = {"obs": {"rgb": {"shape": [3, 240, 240], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
            crop_shape=(216, 216),
            random_crop=True,
            use_seq=False,
        )
        encoder.eval()

        bs = 4
        obs_dict = {"rgb": torch.randn(bs, 3, 240, 240)}

        with torch.no_grad():
            output = encoder(obs_dict)

        assert output.shape == (bs, 256)

    def test_multi_image_encoder_with_resize(self):
        """Test MultiImageObsEncoder with resizing."""
        shape_meta = {"obs": {"rgb": {"shape": [3, 128, 128], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
            resize_shape=(224, 224),
            use_seq=False,
        )
        encoder.eval()

        bs = 4
        obs_dict = {"rgb": torch.randn(bs, 3, 128, 128)}

        with torch.no_grad():
            output = encoder(obs_dict)

        assert output.shape == (bs, 256)

    def test_multi_image_encoder_multiple_rgb_inputs(self):
        """Test MultiImageObsEncoder with multiple RGB inputs."""
        shape_meta = {
            "obs": {
                "rgb1": {"shape": [3, 224, 224], "type": "rgb"},
                "rgb2": {"shape": [3, 224, 224], "type": "rgb"},
            }
        }
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
            share_rgb_model=False,
            use_seq=False,
        )
        encoder.eval()

        bs = 4
        obs_dict = {
            "rgb1": torch.randn(bs, 3, 224, 224),
            "rgb2": torch.randn(bs, 3, 224, 224),
        }

        with torch.no_grad():
            output = encoder(obs_dict)

        assert output.shape == (bs, 256)

    def test_multi_image_encoder_shared_rgb_model(self):
        """Test MultiImageObsEncoder with shared RGB model."""
        shape_meta = {
            "obs": {
                "rgb1": {"shape": [3, 224, 224], "type": "rgb"},
                "rgb2": {"shape": [3, 224, 224], "type": "rgb"},
            }
        }
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
            share_rgb_model=True,
            use_seq=False,
        )
        encoder.eval()

        bs = 4
        obs_dict = {
            "rgb1": torch.randn(bs, 3, 224, 224),
            "rgb2": torch.randn(bs, 3, 224, 224),
        }

        with torch.no_grad():
            output = encoder(obs_dict)

        assert output.shape == (bs, 256)

    def test_multi_image_encoder_with_sequence(self):
        """Test MultiImageObsEncoder with sequence input."""
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
            use_seq=True,
            keep_horizon_dims=False,
        )
        encoder.eval()

        bs = 4
        seq_len = 8
        obs_dict = {"rgb": torch.randn(bs, seq_len, 3, 224, 224)}

        with torch.no_grad():
            output = encoder(obs_dict)

        assert output.shape == (bs, seq_len * 256)

    def test_multi_image_encoder_with_sequence_keep_dims(self):
        """Test MultiImageObsEncoder with sequence input and keep_horizon_dims."""
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
            use_seq=True,
            keep_horizon_dims=True,
        )
        encoder.eval()

        bs = 4
        seq_len = 8
        obs_dict = {"rgb": torch.randn(bs, seq_len, 3, 224, 224)}

        with torch.no_grad():
            output = encoder(obs_dict)

        assert output.shape == (bs, seq_len, 256)


class TestCropFunctions:
    """Test suite for crop utility functions."""

    def test_crop_image_from_indices(self):
        """Test crop_image_from_indices function."""
        bs = 4
        images = torch.randn(bs, 3, 100, 100)
        crop_indices = torch.zeros(bs, 2).long()  # Top-left corner
        crop_height = 50
        crop_width = 50

        crops = crop_image_from_indices(images, crop_indices, crop_height, crop_width)

        assert crops.shape == (bs, 3, 50, 50)

    def test_crop_image_from_indices_multiple_crops(self):
        """Test crop_image_from_indices with multiple crops per image."""
        bs = 2
        num_crops = 3
        images = torch.randn(bs, 3, 100, 100)
        crop_indices = torch.zeros(bs, num_crops, 2).long()
        crop_height = 50
        crop_width = 50

        crops = crop_image_from_indices(images, crop_indices, crop_height, crop_width)

        assert crops.shape == (bs, num_crops, 3, 50, 50)

    def test_sample_random_image_crops(self):
        """Test sample_random_image_crops function."""
        bs = 4
        images = torch.randn(bs, 3, 100, 100)
        crop_height = 50
        crop_width = 50
        num_crops = 2

        crops, crop_inds = sample_random_image_crops(
            images, crop_height, crop_width, num_crops, pos_enc=False
        )

        assert crops.shape == (bs, num_crops, 3, 50, 50)
        assert crop_inds.shape == (bs, num_crops, 2)

    def test_sample_random_image_crops_with_pos_enc(self):
        """Test sample_random_image_crops with position encoding."""
        bs = 4
        images = torch.randn(bs, 3, 100, 100)
        crop_height = 50
        crop_width = 50
        num_crops = 2

        crops, crop_inds = sample_random_image_crops(
            images, crop_height, crop_width, num_crops, pos_enc=True
        )

        assert crops.shape == (bs, num_crops, 5, 50, 50)  # 3 + 2 for position encoding
        assert crop_inds.shape == (bs, num_crops, 2)


class TestGetResnet:
    """Test suite for get_resnet function."""

    def test_get_resnet18(self):
        """Test getting resnet18."""
        model = get_resnet("resnet18")
        assert model is not None
        assert isinstance(model.fc, nn.Identity)

    def test_get_resnet34(self):
        """Test getting resnet34."""
        model = get_resnet("resnet34")
        assert model is not None
        assert isinstance(model.fc, nn.Identity)

    def test_get_resnet50(self):
        """Test getting resnet50."""
        model = get_resnet("resnet50")
        assert model is not None
        assert isinstance(model.fc, nn.Identity)


class TestEncoderIntegration:
    """Test integration of encoders with the MLP network."""

    def test_identity_encoder_with_state_obs(self):
        """Test IdentityEncoder output dimension for state observations."""
        # For state observations, obs_dim should match the actual state dimension
        encoder = IdentityEncoder(dropout=0.0)
        encoder.eval()

        bs = 4
        To = 2
        obs_dim = 20  # state observation dimension

        # Simulate observation sequence
        condition = torch.randn(bs, To, obs_dim)

        with torch.no_grad():
            output = encoder(condition)

        # Output should have same shape as input
        assert output.shape == (bs, To, obs_dim)

    def test_multi_image_encoder_output_dimension(self):
        """Test MultiImageObsEncoder output dimension for image observations."""
        # For image observations, output should be emb_dim (256 by default)
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        emb_dim = 256  # This is the key dimension for MLP integration
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=emb_dim,
            use_seq=True,
            keep_horizon_dims=True,
        )
        encoder.eval()

        bs = 4
        To = 2
        obs_dict = {"rgb": torch.randn(bs, To, 3, 224, 224)}

        with torch.no_grad():
            output = encoder(obs_dict)

        # Output should be (bs, To, emb_dim) where emb_dim=256
        # This is what the MLP expects as single_condition_dim["image"]
        assert output.shape == (bs, To, emb_dim)

    def test_encoder_output_compatibility_with_mlp(self):
        """Test that encoder output is compatible with MLP input expectations."""
        # Test state encoder
        state_encoder = IdentityEncoder(dropout=0.0)
        state_encoder.eval()

        bs = 4
        To = 2
        state_obs_dim = 20
        state_condition = torch.randn(bs, To, state_obs_dim)

        with torch.no_grad():
            state_output = state_encoder(state_condition)

        # For state: obs_dim = state_obs_dim
        assert state_output.shape == (bs, To, state_obs_dim)

        # Test image encoder
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        image_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=256,
            use_seq=True,
            keep_horizon_dims=True,
        )
        image_encoder.eval()

        image_obs_dict = {"rgb": torch.randn(bs, To, 3, 224, 224)}

        with torch.no_grad():
            image_output = image_encoder(image_obs_dict)

        # For image: obs_dim = 256 (emb_dim)
        assert image_output.shape == (bs, To, 256)


class TestEncoderMLPIntegration:
    """Test suite for encoder and MLP integration."""

    def test_identity_encoder_with_vanilla_mlp(self):
        """Test IdentityEncoder with VanillaMLP for state-based observations."""
        # Configuration for state observations
        act_dim = 10
        Ta = 8
        state_obs_dim = 20  # Raw state observation dimension
        To = 2

        # Create encoder for state observations
        encoder = IdentityEncoder(dropout=0.0)
        encoder.eval()

        # Create MLP with obs_dim matching state dimension
        mlp = VanillaMLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=state_obs_dim,  # Use raw state dimension
            To=To,
            emb_dim=512,
            n_layers=6,
            dropout=0.1,
        )
        mlp.eval()

        # Test forward pass
        bs = 4
        actions = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        state_obs = torch.randn(bs, To, state_obs_dim)

        with torch.no_grad():
            # Encode observations (identity encoder keeps same shape)
            encoded_obs = encoder(state_obs)
            assert encoded_obs.shape == (bs, To, state_obs_dim)

            # Pass through MLP
            output, scalar_out = mlp(actions, s, t, encoded_obs)

        assert output.shape == (bs, Ta, act_dim)
        assert scalar_out.shape == (bs, 1)

    def test_image_encoder_with_vanilla_mlp(self):
        """Test MultiImageObsEncoder with VanillaMLP for image-based observations."""
        # Configuration for image observations
        act_dim = 10
        Ta = 8
        image_emb_dim = 256  # Key: encoder output dimension
        To = 2

        # Create encoder for image observations
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=image_emb_dim,
            use_seq=True,
            keep_horizon_dims=True,
        )
        encoder.eval()

        # Create MLP with obs_dim matching encoder output dimension
        mlp = VanillaMLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=image_emb_dim,  # Use encoder output dimension (256)
            To=To,
            emb_dim=512,
            n_layers=6,
            dropout=0.1,
        )
        mlp.eval()

        # Test forward pass
        bs = 4
        actions = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        image_obs = {"rgb": torch.randn(bs, To, 3, 224, 224)}

        with torch.no_grad():
            # Encode observations (encoder outputs embedding)
            encoded_obs = encoder(image_obs)
            assert encoded_obs.shape == (bs, To, image_emb_dim)

            # Pass through MLP
            output, scalar_out = mlp(actions, s, t, encoded_obs)

        assert output.shape == (bs, Ta, act_dim)
        assert scalar_out.shape == (bs, 1)

    def test_identity_encoder_with_mlp(self):
        """Test IdentityEncoder with MLP for state-based observations."""
        # Configuration for state observations
        act_dim = 7
        Ta = 8
        state_obs_dim = 20
        To = 3

        # Create encoder for state observations
        encoder = IdentityEncoder(dropout=0.0)
        encoder.eval()

        # Create MLP with obs_dim matching state dimension
        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=state_obs_dim,
            To=To,
            emb_dim=512,
            n_layers=6,
            timestep_emb_dim=128,
            max_freq=100.0,
            disable_time_embedding=False,
            dropout=0.1,
        )
        mlp.eval()

        # Test forward pass
        bs = 4
        actions = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        state_obs = torch.randn(bs, To, state_obs_dim)

        with torch.no_grad():
            encoded_obs = encoder(state_obs)
            output, scalar_out = mlp(actions, s, t, encoded_obs)

        assert output.shape == (bs, Ta, act_dim)
        assert scalar_out.shape == (bs, 1)

    def test_image_encoder_with_mlp(self):
        """Test MultiImageObsEncoder with MLP for image-based observations."""
        # Configuration for image observations
        act_dim = 7
        Ta = 8
        image_emb_dim = 256
        To = 3

        # Create encoder for image observations
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=image_emb_dim,
            use_seq=True,
            keep_horizon_dims=True,
        )
        encoder.eval()

        # Create MLP with obs_dim matching encoder output dimension
        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=image_emb_dim,
            To=To,
            emb_dim=512,
            n_layers=6,
            timestep_emb_dim=128,
            max_freq=100.0,
            disable_time_embedding=False,
            dropout=0.1,
        )
        mlp.eval()

        # Test forward pass
        bs = 4
        actions = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        image_obs = {"rgb": torch.randn(bs, To, 3, 224, 224)}

        with torch.no_grad():
            encoded_obs = encoder(image_obs)
            output, scalar_out = mlp(actions, s, t, encoded_obs)

        assert output.shape == (bs, Ta, act_dim)
        assert scalar_out.shape == (bs, 1)

    def test_multi_image_encoder_with_mlp(self):
        """Test MultiImageObsEncoder with multiple RGB inputs and MLP."""
        # Configuration for multi-image observations
        act_dim = 7
        Ta = 8
        image_emb_dim = 256
        To = 3

        # Create encoder for multiple image observations
        shape_meta = {
            "obs": {
                "sideview_image": {"shape": [3, 240, 240], "type": "rgb"},
                "eye_in_hand_image": {"shape": [3, 240, 240], "type": "rgb"},
                "eef_pos": {"shape": [3], "type": "low_dim"},
                "eef_quat": {"shape": [4], "type": "low_dim"},
            }
        }
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=image_emb_dim,
            crop_shape=(216, 216),
            random_crop=True,
            use_group_norm=True,
            use_seq=True,
            keep_horizon_dims=True,
        )
        encoder.eval()

        # Create MLP with obs_dim matching encoder output dimension
        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=image_emb_dim,
            To=To,
            emb_dim=512,
            n_layers=6,
            timestep_emb_dim=128,
            dropout=0.1,
        )
        mlp.eval()

        # Test forward pass
        bs = 4
        actions = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        obs_dict = {
            "sideview_image": torch.randn(bs, To, 3, 240, 240),
            "eye_in_hand_image": torch.randn(bs, To, 3, 240, 240),
            "eef_pos": torch.randn(bs, To, 3),
            "eef_quat": torch.randn(bs, To, 4),
        }

        with torch.no_grad():
            encoded_obs = encoder(obs_dict)
            assert encoded_obs.shape == (bs, To, image_emb_dim)

            output, scalar_out = mlp(actions, s, t, encoded_obs)

        assert output.shape == (bs, Ta, act_dim)
        assert scalar_out.shape == (bs, 1)

    def test_encoder_obs_dim_selection(self):
        """Test helper logic for selecting obs_dim based on observation type."""
        # This demonstrates the pattern mentioned in the user query:
        # single_condition_dim = {
        #     "state": args.obs_dim,
        #     "image": 256,
        # }[args.obs_type]

        state_obs_dim = 20
        image_emb_dim = 256

        # Simulate different observation types
        obs_type = "state"
        single_condition_dim = {
            "state": state_obs_dim,
            "image": image_emb_dim,
        }[obs_type]
        assert single_condition_dim == state_obs_dim

        obs_type = "image"
        single_condition_dim = {
            "state": state_obs_dim,
            "image": image_emb_dim,
        }[obs_type]
        assert single_condition_dim == image_emb_dim

    def test_end_to_end_state_pipeline(self):
        """Test complete pipeline with state observations."""
        # Setup
        obs_type = "state"
        act_dim = 10
        Ta = 8
        state_obs_dim = 20
        To = 2

        # Determine condition dimension based on obs_type
        single_condition_dim = {
            "state": state_obs_dim,
            "image": 256,
        }[obs_type]

        # Create encoder
        encoder = IdentityEncoder(dropout=0.0)
        encoder.eval()

        # Create MLP
        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=single_condition_dim,
            To=To,
            emb_dim=512,
            n_layers=6,
        )
        mlp.eval()

        # Test
        bs = 4
        actions = torch.randn(bs, Ta, act_dim)
        s = torch.randn(bs)
        t = torch.randn(bs)
        obs = torch.randn(bs, To, state_obs_dim)

        with torch.no_grad():
            encoded = encoder(obs)
            output, scalar_out = mlp(actions, s, t, encoded)

        assert output.shape == (bs, Ta, act_dim)
        assert scalar_out.shape == (bs, 1)

    def test_end_to_end_image_pipeline(self):
        """Test complete pipeline with image observations."""
        # Setup
        obs_type = "image"
        act_dim = 10
        Ta = 8
        To = 2

        # Determine condition dimension based on obs_type
        single_condition_dim = {
            "state": 20,  # Would be args.obs_dim if state
            "image": 256,
        }[obs_type]

        # Create encoder
        shape_meta = {"obs": {"rgb": {"shape": [3, 224, 224], "type": "rgb"}}}
        encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model_name="resnet18",
            emb_dim=single_condition_dim,
            use_seq=True,
            keep_horizon_dims=True,
        )
        encoder.eval()

        # Create MLP
        mlp = MLP(
            act_dim=act_dim,
            Ta=Ta,
            obs_dim=single_condition_dim,
            To=To,
            emb_dim=512,
            n_layers=6,
        )
        mlp.eval()

        # Test
        bs = 4
        torch.randn(bs, Ta, act_dim)
        torch.randn(bs)
        torch.randn(bs)
        obs_dict = {"rgb": torch.randn(bs, To, 3, 224, 224)}

        with torch.no_grad():
            encoded = encoder(obs_dict)
            assert encoded.shape == (bs, To, single_condition_dim)
