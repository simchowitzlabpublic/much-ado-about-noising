"""Observation encoder.

Port from https://github.com/CleanDiffuserTeam/CleanDiffuser

Author: Chaoyi Pan
Date: 2025-10-03
"""

import copy
from collections.abc import Callable

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as ttf

import mip.torch_utils as tu
from mip.torch_utils import at_least_ndim


def get_mask(
    mask: torch.Tensor,
    mask_shape: tuple,
    dropout: float,
    train: bool,
    device: torch.device,
):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1.0 if mask is None else mask
    return mask


class CropRandomizer(nn.Module):
    """Randomly sample crops at input, and then average across crop features at output."""

    def __init__(
        self,
        input_shape,
        crop_height,
        crop_width,
        num_crops=1,
        pos_enc=False,
    ):
        """Args:
        input_shape (tuple, list): shape of input (not including batch dimension)
        crop_height (int): crop height
        crop_width (int): crop width
        num_crops (int): number of random crops to take
        pos_enc (bool): if True, add 2 channels to the output to encode the spatial
            location of the cropped pixels in the source image.
        """
        super().__init__()

        assert len(input_shape) == 3  # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3  # must have at least (C, H, W) dimensions
        if self.training:
            # generate random crops
            out, _ = sample_random_image_crops(
                images=inputs,
                crop_height=self.crop_height,
                crop_width=self.crop_width,
                num_crops=self.num_crops,
                pos_enc=self.pos_enc,
            )
            # [B, N, ...] -> [B * N, ...]
            return tu.join_dimensions(out, 0, 1)
        else:
            # take center crop during eval
            out = ttf.center_crop(
                img=inputs, output_size=(self.crop_height, self.crop_width)
            )
            if self.num_crops > 1:
                B, C, H, W = out.shape
                out = (
                    out.unsqueeze(1)
                    .expand(B, self.num_crops, C, H, W)
                    .reshape(-1, C, H, W)
                )
                # [B * N, ...]
            return out

    def forward_out(self, inputs):
        """Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        if self.num_crops <= 1:
            return inputs
        else:
            batch_size = inputs.shape[0] // self.num_crops
            out = tu.reshape_dimensions(
                inputs,
                begin_axis=0,
                end_axis=0,
                target_dims=(batch_size, self.num_crops),
            )
            return out.mean(dim=1)

    def forward(self, inputs):
        return self.forward_in(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = f"{str(self.__class__.__name__)}"
        msg = (
            header
            + f"(input_shape={self.input_shape}, crop_size=[{self.crop_height}, {self.crop_width}], num_crops={self.num_crops})"
        )
        return msg


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    """Crops images at the locations specified by @crop_indices. Crops will be
    taken across all channels.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_indices (torch.Tensor): batch of indices of shape [..., N, 2] where
            N is the number of crops to take per image and each entry corresponds
            to the pixel height and width of where to take the crop. Note that
            the indices can also be of shape [..., 2] if only 1 crop should
            be taken per image. Leading dimensions must be consistent with
            @images argument. Each index specifies the top left of the crop.
            Values must be in range [0, H - CH - 1] x [0, W - CW - 1] where
            H and W are the height and width of @images and CH and CW are
            @crop_height and @crop_width.

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

    Returns:
        crops (torch.Tesnor): cropped images of shape [..., C, @crop_height, @crop_width]
    """
    # make sure length of input shapes is consistent
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (
        ndim_im_shape == ndim_indices_shape + 2
    )

    # maybe pad so that @crop_indices is shape [..., N, 2]
    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    # make sure leading dimensions between images and indices are consistent
    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_c, image_h, image_w = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    # make sure @crop_indices are in valid range
    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] < (image_h - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] < (image_w - crop_width)).all().item()

    # convert each crop index (ch, cw) into a list of pixel indices that correspond to the entire window.

    # 2D index array with columns [0, 1, ..., CH - 1] and shape [CH, CW]
    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = tu.unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    # 2D index array with rows [0, 1, ..., CW - 1] and shape [CH, CW]
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = tu.unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    # combine into shape [CH, CW, 2]
    crop_in_grid = torch.cat(
        (crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)), dim=-1
    )

    # Add above grid with the offset index of each sampled crop to get 2d indices for each crop.
    # After broadcasting, this will be shape [..., N, CH, CW, 2] and each crop has a [CH, CW, 2]
    # shape array that tells us which pixels from the corresponding source image to grab.
    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(
        grid_reshape
    )

    # For using @torch.gather, convert to flat indices from 2D indices, and also
    # repeat across the channel dimension. To get flat index of each pixel to grab for
    # each sampled crop, we just use the mapping: ind = h_ind * @image_w + w_ind
    all_crop_inds = (
        all_crop_inds[..., 0] * image_w + all_crop_inds[..., 1]
    )  # shape [..., N, CH, CW]
    all_crop_inds = tu.unsqueeze_expand_at(
        all_crop_inds, size=image_c, dim=-3
    )  # shape [..., N, C, CH, CW]
    all_crop_inds = tu.flatten(
        all_crop_inds, begin_axis=-2
    )  # shape [..., N, C, CH * CW]

    # Repeat and flatten the source images -> [..., N, C, H * W] and then use gather to index with crop pixel inds
    images_to_crop = tu.unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = tu.flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    # [..., N, C, CH * CW] -> [..., N, C, CH, CW]
    reshape_axis = len(crops.shape) - 1
    crops = tu.reshape_dimensions(
        crops,
        begin_axis=reshape_axis,
        end_axis=reshape_axis,
        target_dims=(crop_height, crop_width),
    )

    if is_padded:
        # undo padding -> [..., C, CH, CW]
        crops = crops.squeeze(-4)
    return crops


def sample_random_image_crops(
    images, crop_height, crop_width, num_crops, pos_enc=False
):
    """For each image, randomly sample @num_crops crops of size (@crop_height, @crop_width), from
    @images.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

        num_crops (n): number of crops to sample

        pos_enc (bool): if True, also add 2 channels to the outputs that gives a spatial
            encoding of the original source pixel locations. This means that the
            output crops will contain information about where in the source image
            it was sampled from.

    Returns:
        crops (torch.Tensor): crops of shape (..., @num_crops, C, @crop_height, @crop_width)
            if @pos_enc is False, otherwise (..., @num_crops, C + 2, @crop_height, @crop_width)

        crop_inds (torch.Tensor): sampled crop indices of shape (..., N, 2)
    """
    device = images.device

    # maybe add 2 channels of spatial encoding to the source image
    source_im = images
    if pos_enc:
        # spatial encoding [y, x] in [0, 1]
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos_y = pos_y.float().to(device) / float(h)
        pos_x = pos_x.float().to(device) / float(w)
        position_enc = torch.stack((pos_y, pos_x))  # shape [C, H, W]

        # unsqueeze and expand to match leading dimensions -> shape [..., C, H, W]
        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)

        # concat across channel dimension with input
        source_im = torch.cat((source_im, position_enc), dim=-3)

    # make sure sample boundaries ensure crops are fully within the images
    image_c, image_h, image_w = source_im.shape[-3:]
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width

    # Sample crop locations for all tensor dimensions up to the last 3, which are [C, H, W].
    # Each gets @num_crops samples - typically this will just be the batch dimension (B), so
    # we will sample [B, N] indices, but this supports having more than one leading dimension,
    # or possibly no leading dimension.
    #
    # Trick: sample in [0, 1) with rand, then re-scale to [0, M) and convert to long to get sampled ints
    crop_inds_h = (
        max_sample_h * torch.rand(*source_im.shape[:-3], num_crops).to(device)
    ).long()
    crop_inds_w = (
        max_sample_w * torch.rand(*source_im.shape[:-3], num_crops).to(device)
    ).long()
    crop_inds = torch.cat(
        (crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)), dim=-1
    )  # shape [..., N, 2]

    crops = crop_image_from_indices(
        images=source_im,
        crop_indices=crop_inds,
        crop_height=crop_height,
        crop_width=crop_width,
    )

    return crops, crop_inds


class BaseEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        raise NotImplementedError


class IdentityEncoder(BaseEncoder):
    """Identity encoder does not change the input condition.

    Input:
        - condition: (b, *cond_in_shape)
        - mask :     (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_in_shape)
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        mask = at_least_ndim(
            get_mask(
                mask,
                (condition.shape[0],),
                self.dropout,
                self.training,
                condition.device,
            ),
            condition.dim(),
        )
        return condition * mask


class Mlp(nn.Module):
    """**Multilayer perceptron.** A simple pytorch MLP module.

    Args:
        in_dim: int,
            The dimension of the input tensor.
        hidden_dims: List[int],
            A list of integers, each element is the dimension of the hidden layer.
        out_dim: int,
            The dimension of the output tensor.
        activation: nn.Module,
            The activation function used in the hidden layers.
        out_activation: nn.Module,
            The activation function used in the output layer.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        activation: nn.Module = nn.ReLU(),
        out_activation: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(in_dim if i == 0 else hidden_dims[i - 1], hidden_dims[i]),
                    activation,
                )
                for i in range(len(hidden_dims))
            ],
            nn.Linear(hidden_dims[-1], out_dim),
            out_activation,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)


class MLPEncoder(BaseEncoder):
    """A simple MLP encoder.

    Use a simple MLP to project the input condition to the desired dimension.

    Args:
        in_dim: int,
            The input dimension of the condition
        out_dim: int,
            The output dimension of the condition
        hidden_dims: List[int],
            The hidden dimensions of the MLP
        act: nn.Module,
            The activation function of the MLP
        dropout: float,
            The label dropout rate

    Examples:
        >>> encoder = MLPEncoder(in_dim=5, out_dim=10)
        >>> condition = torch.randn(2, 5)
        >>> encoder(condition).shape
        torch.Size([2, 10])
        >>> condition = torch.randn(2, 20, 5)
        >>> encoder(condition).shape
        torch.Size([2, 20, 10])
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: list[int],
        act=nn.LeakyReLU(),
        dropout: float = 0.25,
    ):
        super().__init__(dropout)
        hidden_dims = (
            [
                hidden_dims,
            ]
            if isinstance(hidden_dims, int)
            else hidden_dims
        )
        self.mlp = Mlp(in_dim, hidden_dims, out_dim, act)

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        mask = at_least_ndim(
            get_mask(
                mask,
                (condition.shape[0],),
                self.dropout,
                self.training,
                condition.device,
            ),
            condition.dim(),
        )
        return self.mlp(condition) * mask


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    assert len(bn_list) == 0
    return root_module


def get_resnet(name, weights=None, **kwargs):
    """name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m".
    """
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


class MultiImageObsEncoder(BaseEncoder):
    """Input:
        - condition: {"cond1": (b, *cond1_shape), "cond2": (b, *cond2_shape), ...} or (b, *cond_in_shape)
        - mask :     (b, *mask_shape) or None, None means no mask.

    Output:
        - condition: (b, *cond_out_shape)

    Assumes rgb input: B, C, H, W or B, seq_len, C,H,W
    Assumes low_dim input: B, D or B, seq_len, D
    """

    def __init__(
        self,
        shape_meta: dict,
        rgb_model_name: str,
        emb_dim: int = 256,
        resize_shape: tuple[int, int] | dict[str, tuple] | None = None,
        crop_shape: tuple[int, int] | dict[str, tuple] | None = None,
        random_crop: bool = True,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
        # use_seq: B, seq_len, C, H, W or B, C, H, W
        use_seq=False,
        # if True: (bs, seq_len, embed_dim)
        keep_horizon_dims=False,
    ):
        super().__init__()
        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = {}

        # rgb_model
        if "resnet" in rgb_model_name:
            rgb_model = get_resnet(rgb_model_name)
        else:
            raise ValueError("Fatal rgb_model")

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            # print(key, attr)
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

                this_transform = nn.Sequential(
                    this_resizer, this_randomizer, this_normalizer
                )
                key_transform_map[key] = this_transform
            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        self.use_seq = use_seq
        self.keep_horizon_dims = keep_horizon_dims
        self.mlp = nn.Sequential(
            nn.Linear(self.output_shape(), emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def multi_image_forward(self, obs_dict):
        batch_size = None
        features = []

        if self.use_seq:
            # input: (bs, horizon, c, h, w)
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].flatten(end_dim=1)

        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map["rgb"](imgs)
            # (N,B,D)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature, 0, 1)
            # (B,N*D)
            feature = feature.reshape(batch_size, -1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        features = torch.cat(features, dim=-1)
        return features

    def forward(self, obs_dict, mask=None):
        ori_batch_size, ori_seq_len = self.get_batch_size(obs_dict)
        features = self.multi_image_forward(obs_dict)
        # linear embedding
        result = self.mlp(features)
        if self.use_seq:
            if self.keep_horizon_dims:
                result = result.reshape(ori_batch_size, ori_seq_len, -1)
            else:
                result = result.reshape(ori_batch_size, -1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = {}
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            prefix = (batch_size, 1) if self.use_seq else (batch_size,)
            this_obs = torch.zeros(prefix + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.multi_image_forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape[0]

    def get_batch_size(self, obs_dict):
        any_key = next(iter(obs_dict))
        any_tensor = obs_dict[any_key]
        return any_tensor.size(0), any_tensor.size(1)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
