"""Utility functions for loading and processing datasets.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import functools
import math
import numbers
import os
from collections.abc import Callable
from functools import cached_property

import numba
import numcodecs
import numpy as np
import torch
import zarr
from loguru import logger
from scipy import interpolate

from mip.config import TaskConfig
from mip.datasets import rotation_conversion as rc
from mip.datasets.base import BaseDataset


def get_dataset(config: TaskConfig) -> BaseDataset:
    raise NotImplementedError("Dataset not implemented.")


def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0


def rechunk_recompress_array(
    group, name, chunks=None, chunk_length=None, compressor=None, tmp_key="_temp"
):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)

    if compressor is None:
        compressor = old_arr.compressor

    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr


def get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6, max_chunk_length=None):
    """Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C.
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if (
            this_chunk_bytes <= target_chunk_bytes
            and next_chunk_bytes > target_chunk_bytes
        ):
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(
        this_max_chunk_length, math.ceil(target_chunk_bytes / item_chunk_bytes)
    )
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """

    def __init__(self, root: zarr.Group | dict[str, dict]):
        """Dummy constructor. Use copy_from* and create_from* class methods instead."""
        assert "data" in root
        assert "meta" in root
        assert "episode_ends" in root["meta"]
        # Handle both zarr v3 (no .items()) and dict
        data_group = root["data"]
        if hasattr(data_group, "items"):
            # dict-like interface
            for key, value in data_group.items():
                assert value.shape[0] == root["meta"]["episode_ends"][-1]
        else:
            # zarr v3 Group interface
            for key in data_group:
                value = data_group[key]
                assert value.shape[0] == root["meta"]["episode_ends"][-1]
        self.root = root

    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        root.require_group("data", overwrite=False)
        meta = root.require_group("meta", overwrite=False)
        if "episode_ends" not in meta:
            meta.zeros(
                "episode_ends",
                shape=(0,),
                dtype=np.int64,
                compressor=None,
                overwrite=False,
            )
        return cls(root=root)

    @classmethod
    def create_empty_numpy(cls):
        root = {
            "data": {},
            "meta": {"episode_ends": np.zeros((0,), dtype=np.int64)},
        }
        return cls(root=root)

    @classmethod
    def create_from_group(cls, group, **kwargs):
        if "data" not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode="r", **kwargs):
        """Open an on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)

    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(
        cls,
        src_store,
        store=None,
        keys=None,
        chunks: dict[str, tuple] = None,
        compressors: dict | str | numcodecs.abc.Codec = None,
        if_exists="replace",
        **kwargs,
    ):
        """Load to memory."""
        if compressors is None:
            compressors = {}
        if chunks is None:
            chunks = {}
        src_root = zarr.open_group(store=src_store, mode="r")
        root = None
        if store is None:
            # numpy backend
            meta = {}
            meta_group = src_root["meta"]
            for key in meta_group:
                value = meta_group[key]
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = list(src_root["data"].keys())
            data = {}
            for key in keys:
                arr = src_root["data"][key]
                data[key] = arr[:]

            root = {"meta": meta, "data": data}
        else:
            root = zarr.open_group(store=store, mode="a")
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=src_store,
                dest=store,
                source_path="/meta",
                dest_path="/meta",
                if_exists=if_exists,
            )
            data_group = root.create_group("data", overwrite=True)
            if keys is None:
                keys = src_root["data"].keys()
            for key in keys:
                value = src_root["data"][key]
                cks = cls._resolve_array_chunks(chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value
                )
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value,
                        dest=data_group,
                        name=key,
                        chunks=cks,
                        compressor=cpr,
                        if_exists=if_exists,
                    )
        buffer = cls(root=root)
        return buffer

    @classmethod
    def copy_from_path(
        cls,
        zarr_path,
        backend=None,
        store=None,
        keys=None,
        chunks: dict[str, tuple] = None,
        compressors: dict | str | numcodecs.abc.Codec = None,
        if_exists="replace",
        **kwargs,
    ):
        """Copy a on-disk zarr to in-memory compressed.
        Recommended.
        """
        if compressors is None:
            compressors = {}
        if chunks is None:
            chunks = {}
        if backend == "numpy":
            logger.warning("backend argument is deprecated!")
            store = None
        group = zarr.open(store=os.path.expanduser(zarr_path), mode="r")
        return cls.copy_from_store(
            src_store=group.store,
            store=store,
            keys=keys,
            chunks=chunks,
            compressors=compressors,
            if_exists=if_exists,
            **kwargs,
        )

    # ============= save methods ===============
    def save_to_store(
        self,
        store,
        chunks: dict[str, tuple] | None = None,
        compressors: str | numcodecs.abc.Codec | dict = None,
        if_exists="replace",
        **kwargs,
    ):
        if compressors is None:
            compressors = {}
        if chunks is None:
            chunks = {}
        root = zarr.group(store)
        if self.backend == "zarr":
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store,
                dest=store,
                source_path="/meta",
                dest_path="/meta",
                if_exists=if_exists,
            )
        else:
            meta_group = root.create_group("meta", overwrite=True)
            # save meta, no chunking
            for key, value in self.root["meta"].items():
                _ = meta_group.array(
                    name=key, data=value, shape=value.shape, chunks=value.shape
                )

        # save data, chunk
        data_group = root.create_group("data", overwrite=True)
        for key, value in self.root["data"].items():
            cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value
            )
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = "/data/" + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store,
                        dest=store,
                        source_path=this_path,
                        dest_path=this_path,
                        if_exists=if_exists,
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value,
                        dest=data_group,
                        name=key,
                        chunks=cks,
                        compressor=cpr,
                        if_exists=if_exists,
                    )
            else:
                # numpy
                _ = data_group.array(name=key, data=value, chunks=cks, compressor=cpr)
        return store

    def save_to_path(
        self,
        zarr_path,
        chunks: dict[str, tuple] | None = None,
        compressors: str | numcodecs.abc.Codec | dict = None,
        if_exists="replace",
        **kwargs,
    ):
        if compressors is None:
            compressors = {}
        if chunks is None:
            chunks = {}
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(
            store, chunks=chunks, compressors=compressors, if_exists=if_exists, **kwargs
        )

    @staticmethod
    def resolve_compressor(compressor="default"):
        if compressor == "default":
            compressor = numcodecs.Blosc(
                cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
            )
        elif compressor == "disk":
            compressor = numcodecs.Blosc(
                "zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE
            )
        return compressor

    @classmethod
    def _resolve_array_compressor(
        cls, compressors: dict | str | numcodecs.abc.Codec, key, array
    ):
        # allows compressor to be explicitly set to None
        cpr = "nil"
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == "nil":
            cpr = cls.resolve_compressor("default")
        return cpr

    @classmethod
    def _resolve_array_chunks(cls, chunks: dict | tuple, key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks

    # ============= properties =================
    @cached_property
    def data(self):
        return self.root["data"]

    @cached_property
    def meta(self):
        return self.root["meta"]

    def update_meta(self, data):
        # sanitize data
        np_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == "zarr":
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value,
                    shape=value.shape,
                    chunks=value.shape,
                    overwrite=True,
                )
        else:
            meta_group.update(np_data)

        return meta_group

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    def get_episode_idxs(self):
        import numba

        numba.jit(nopython=True)

        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result

        return _get_episode_idxs(self.episode_ends)

    @property
    def backend(self):
        backend = "numpy"
        if isinstance(self.root, zarr.Group):
            backend = "zarr"
        return backend

    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == "zarr":
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        # Handle zarr arrays which don't have len()
        episode_ends = self.episode_ends
        if hasattr(episode_ends, "shape"):
            # Zarr array or numpy array
            if episode_ends.shape[0] == 0:
                return 0
        elif len(episode_ends) == 0:
            # List or regular array
            return 0
        return int(episode_ends[-1])

    @property
    def n_episodes(self):
        # Handle zarr arrays which don't have len()
        episode_ends = self.episode_ends
        if hasattr(episode_ends, "shape"):
            # Zarr array or numpy array
            return episode_ends.shape[0]
        else:
            # List or regular array
            return len(episode_ends)

    @property
    def chunk_size(self):
        if self.backend == "zarr":
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(
        self,
        data: dict[str, np.ndarray],
        chunks: dict[str, tuple] | None = None,
        compressors: str | numcodecs.abc.Codec | dict = None,
    ):
        if compressors is None:
            compressors = {}
        if chunks is None:
            chunks = {}
        assert len(data) > 0
        is_zarr = self.backend == "zarr"

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert len(value.shape) >= 1
            if episode_length is None:
                episode_length = len(value)
            else:
                assert episode_length == len(value)
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value
                    )
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value
                    )
                    arr = self.data.zeros(
                        name=key,
                        shape=new_shape,
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr,
                    )
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert value.shape[1:] == arr.shape[1:]
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0] :] = value

        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr and episode_ends.chunks[0] < episode_ends.shape[0]:
            rechunk_recompress_array(
                self.meta,
                "episode_ends",
                chunk_length=int(episode_ends.shape[0] * 1.5),
            )

    def drop_episode(self):
        is_zarr = self.backend == "zarr"
        episode_ends = self.episode_ends[:].copy()
        assert len(episode_ends) > 0
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for _key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends) - 1)
        else:
            self.episode_ends.resize(len(episode_ends) - 1, refcheck=False)

    def pop_episode(self):
        assert self.n_episodes > 0
        episode = self.get_episode(self.n_episodes - 1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result

    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx - 1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = {}
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result

    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == "zarr"
        chunks = {}
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks

    def set_chunks(self, chunks: dict):
        assert self.backend == "zarr"
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == "zarr"
        compressors = {}
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == "zarr"
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)


# -----------------------------------------------------------------------------#
# ------------------------------ SequenceSampler ------------------------------#
# -----------------------------------------------------------------------------#

# Original implemetation: https://github.com/real-stanford/diffusion_policy
# Observation Horizon: To|n_obs_steps
# Action Horizon: Ta|n_action_steps
# Prediction Horizon: T|horizon
# To = 3
# Ta = 4
# T = 6
# |o|o|o|
# | | |a|a|a|a|
# pad_before = 2
# pad_after = 3


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0  # episode start index
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]  # episode end index
        episode_length = end_idx - start_idx  # episode length

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (
                    buffer_end_idx - buffer_start_idx
                )
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


class SequenceSampler:
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=None,
        zero_padding: bool = False,
    ):
        """key_first_k: dict str: int
        Only take first k data from these keys (to improve perf).
        """
        if key_first_k is None:
            key_first_k = {}
        super().__init__()
        assert sequence_length >= 1

        # all keys
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]

        # create indices
        # indices (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        # buffer_start_idx and buffer_end_idx define the actual start and end positions of the sample sequence within the original dataset.
        # sample_start_idx and sample_end_idx define the relative start and end positions within the sample sequence,
        # which is particularly useful when dealing with padding as it can affect the actual length of the sequence.
        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.zero_padding = zero_padding
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )
        result = {}
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full(
                    (n_data,) + input_arr.shape[1:],
                    fill_value=np.nan,
                    dtype=input_arr.dtype,
                )
                sample[:k_data] = input_arr[
                    buffer_start_idx : buffer_start_idx + k_data
                ]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                if not self.zero_padding:
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result


# -----------------------------------------------------------------------------#
# ---------------------------- Rotation Transformer ---------------------------#
# -----------------------------------------------------------------------------#


class RotationTransformer:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        """Valid representations.

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = []
        inverse_funcs = []

        if from_rep != "matrix":
            funcs = [
                getattr(rc, f"{from_rep}_to_matrix"),
                getattr(rc, f"matrix_to_{from_rep}"),
            ]
            if from_convention is not None:
                funcs = [
                    functools.partial(func, convention=from_convention)
                    for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [
                getattr(rc, f"matrix_to_{to_rep}"),
                getattr(rc, f"{to_rep}_to_matrix"),
            ]
            if to_convention is not None:
                funcs = [
                    functools.partial(func, convention=to_convention) for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(
        x: np.ndarray | torch.Tensor, funcs: list
    ) -> np.ndarray | torch.Tensor:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.tensor(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self._apply_funcs(x, self.inverse_funcs)


# -----------------------------------------------------------------------------#
# --------------------------- multi-field normalizer --------------------------#
# -----------------------------------------------------------------------------#


def empirical_cdf(sample):
    """https://stackoverflow.com/a/33346366."""
    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob


class CDFNormalizer1d:
    """CDF normalizer for a single dimension."""

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)
        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        y = self.fn(x)
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        x = (x + 1) / 2.0
        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            logger.warning(
                f"""[ dataset/normalization ] Warning: out of range in unnormalize: """
                f"""[{x.min()}, {x.max()}] | """
                f"""x : [{self.xmin}, {self.xmax}] | """
                f"""y: [{self.ymin}, {self.ymax}]"""
            )
        x = np.clip(x, self.ymin, self.ymax)
        y = self.inv(x)
        return y


class CDFNormalizer:
    """makes training data uniform (over each dimension) by transforming it with marginal CDFs."""

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins, self.maxs = X.min(0), X.max(0)
        self.dim = X.shape[-1]
        self.cdfs = [CDFNormalizer1d(self.X[:, i]) for i in range(self.dim)]

    def wrap(self, fn_name, x):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap("normalize", x)

    def unnormalize(self, x):
        return self.wrap("unnormalize", x)


class GaussianNormalizer:
    """normalizes data to have zero mean and unit variance."""

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.means, self.stds = X.mean(0), X.std(0)
        self.stds[self.stds == 0] = 1.0

    def normalize(self, x):
        return (x - self.means[None,]) / self.stds[None,]

    def unnormalize(self, x):
        return x * self.stds[None,] + self.means[None,]


class ImageNormalizer:
    """normalizes image data from range [0, 1] to [-1, 1]."""

    def __init__(self):
        pass

    def normalize(self, x):
        return x * 2.0 - 1.0

    def unnormalize(self, x):
        return (x + 1.0) / 2.0


class MinMaxNormalizer:
    """normalizes data through maximum and minimum expansion."""

    def __init__(self, X):
        X = X.reshape(-1, X.shape[-1]).astype(np.float32)
        self.min, self.max = np.min(X, axis=0), np.max(X, axis=0)
        self.range = self.max - self.min
        if np.any(self.range == 0):
            self.range = self.max - self.min
            logger.warning(
                "Warning: Some features have the same min and max value. These will be set to 0."
            )
            self.range[self.range == 0] = 1

    def normalize(self, x):
        x = x.astype(np.float32)
        # nomalize to [0,1]
        nx = (x - self.min) / self.range
        # normalize to [-1, 1]
        nx = nx * 2 - 1
        return nx

    def unnormalize(self, x):
        x = x.astype(np.float32)
        nx = (x + 1) / 2
        x = nx * self.range + self.min
        return x


class EmptyNormalizer:
    """do nothing and change nothing."""

    def __init__(self):
        pass

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


class IdentityNormalizer:
    """Pass-through normalizer that leaves data unchanged."""

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


class CompositeNormalizer:
    """Applies different normalizers to different dimension slices.

    Example: for 10D actions [pos(3) + rot6d(6) + grip(1)]:
        CompositeNormalizer(
            normalizers=[MinMaxNormalizer(pos_data), IdentityNormalizer(), MinMaxNormalizer(grip_data)],
            dim_slices=[(0, 3), (3, 9), (9, 10)],
        )
    """

    def __init__(self, normalizers: list, dim_slices: list[tuple[int, int]]):
        assert len(normalizers) == len(dim_slices)
        self.normalizers = normalizers
        self.dim_slices = dim_slices

    def normalize(self, x):
        out = np.empty_like(x, dtype=np.float32)
        for norm, (start, end) in zip(self.normalizers, self.dim_slices, strict=True):
            out[..., start:end] = norm.normalize(x[..., start:end])
        return out

    def unnormalize(self, x):
        out = np.empty_like(x, dtype=np.float32)
        for norm, (start, end) in zip(self.normalizers, self.dim_slices, strict=True):
            out[..., start:end] = norm.unnormalize(x[..., start:end])
        return out


# -----------------------------------------------------------------------------#
# ------------------------------- useful tool ---------------------------------#
# -----------------------------------------------------------------------------#


def dict_apply(
    x: dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result


def loop_dataloader(dl):
    while True:
        yield from dl


def loop_two_dataloaders(dl1, dl2):
    while True:
        yield from zip(dl1, dl2, strict=False)
