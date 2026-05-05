"""AsyncVectorEnv subclass that accepts a ``dummy_env_fn`` for space probing.

Gymnasium's ``AsyncVectorEnv.__init__`` probes ``observation_space`` /
``action_space`` by instantiating ``env_fns[0]()`` in the parent process. For
MuJoCo-rendered envs (e.g. image-based robomimic) that creates an OpenGL context
in the parent; after ``fork()``, the child's own context creation conflicts.
Passing a ``dummy_env_fn`` that builds a no-render env lets the parent read
spaces without touching GL, while each forked worker creates exactly one
context via its real ``env_fn``. Same trick as sim_env's vendored fork.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Sequence
from multiprocessing import Queue
from multiprocessing.connection import Connection

import numpy as np
from gymnasium import Space
from gymnasium.core import Env
from gymnasium.error import CustomSpaceError
from gymnasium.vector import AsyncVectorEnv as _GymAsyncVectorEnv
from gymnasium.vector.async_vector_env import AsyncState, _async_worker
from gymnasium.vector.utils import (
    CloudpickleWrapper,
    batch_differing_spaces,
    batch_space,
    clear_mpi_env_vars,
    create_empty_array,
    create_shared_memory,
    read_from_shared_memory,
)
from gymnasium.vector.vector_env import AutoresetMode


class AsyncVectorEnv(_GymAsyncVectorEnv):
    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env]],
        dummy_env_fn: Callable[[], Env] | None = None,
        shared_memory: bool = True,
        copy: bool = True,
        context: str | None = None,
        daemon: bool = True,
        worker: (
            Callable[
                [int, Callable[[], Env], Connection, Connection, bool, Queue], None
            ]
            | None
        ) = None,
        observation_mode: str | Space = "same",
        autoreset_mode: str | AutoresetMode = AutoresetMode.NEXT_STEP,
    ):
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        self.context = context
        self.daemon = daemon
        self.worker = worker
        self.observation_mode = observation_mode
        self.autoreset_mode = (
            autoreset_mode
            if isinstance(autoreset_mode, AutoresetMode)
            else AutoresetMode(autoreset_mode)
        )

        self.num_envs = len(env_fns)

        dummy_env = (dummy_env_fn or env_fns[0])()

        self.metadata = dummy_env.metadata
        self.metadata["autoreset_mode"] = self.autoreset_mode
        self.render_mode = dummy_env.render_mode

        self.single_action_space = dummy_env.action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        if isinstance(observation_mode, tuple) and len(observation_mode) == 2:
            assert isinstance(observation_mode[0], Space)
            assert isinstance(observation_mode[1], Space)
            self.observation_space, self.single_observation_space = observation_mode
        else:
            if observation_mode == "same":
                self.single_observation_space = dummy_env.observation_space
                self.observation_space = batch_space(
                    self.single_observation_space, self.num_envs
                )
            elif observation_mode == "different":
                env_spaces = [env().observation_space for env in self.env_fns]
                self.single_observation_space = env_spaces[0]
                self.observation_space = batch_differing_spaces(env_spaces)
            else:
                raise ValueError(
                    f"Invalid `observation_mode`, expected: 'same' or 'different' or tuple of single and batch observation space, actual got {observation_mode}"
                )

        dummy_env.close()
        del dummy_env

        ctx = multiprocessing.get_context(context)
        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    self.single_observation_space, _obs_buffer, n=self.num_envs
                )
            except CustomSpaceError as e:
                raise ValueError(
                    "Using `AsyncVector(..., shared_memory=True)` caused an error, you can disable this feature with `shared_memory=False` however this is slower."
                ) from e
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = worker or _async_worker
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker<{type(self).__name__}>-{idx}",
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                        self.autoreset_mode,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_spaces()
