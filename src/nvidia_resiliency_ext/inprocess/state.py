# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import enum
import os
from typing import Optional


class Mode(enum.Enum):
    r'''
    Indicates operational mode of the current distributed rank.

    Attributes:
        INITIALIZED: the :py:class:`State` was initialized,
            :py:class:`RankAssignment` was not yet performed
        ACTIVE: the rank calls the wrapped function
        INACTIVE: the rank is waiting idle
        TERMINATED: the rank was terminated
    '''

    INITIALIZED = enum.auto()
    ACTIVE = enum.auto()
    INACTIVE = enum.auto()
    TERMINATED = enum.auto()


@dataclasses.dataclass
class State:
    r'''
    Represents the current state of the :py:class:`inprocess.Wrapper`.

    Args:
        rank: a distributed rank index as seen by the
            :py:class:`inprocess.Wrapper`, :py:obj:`None` for terminated ranks
        world_size: a total number of distributed ranks controlled by the
            :py:class:`inprocess.Wrapper`
        active_rank: a distributed rank index passed to the wrapped function
        active_world_size: a total number of distributed ranks passed to the
            wrapped function
        initial_rank: an distributed rank index, captured when the
            :py:class:`Wrapper` was invoked
        initial_world_size: a total number of initial distributed ranks
        iteration: index of the current restart iteration (can be local or global)
        use_global_iteration: whether to use globally unique iteration numbers
        mode: operational mode
        fn_exception: an instance of :py:exc:`Exception` raised by the wrapped
            function in the current restart iteration, :py:obj:`None` if no
            exception was raised
    '''

    rank: Optional[int]
    world_size: int

    active_rank: Optional[int] = None
    active_world_size: Optional[int] = None

    initial_rank: Optional[int] = None
    initial_world_size: Optional[int] = None

    iteration: int = 0
    use_global_iteration: bool = False
    mode: Mode = Mode.INITIALIZED
    fn_exception: Optional[Exception] = None

    def __post_init__(self):
        if self.initial_rank is None:
            self.initial_rank = self.rank
        if self.initial_world_size is None:
            self.initial_world_size = self.world_size

    @classmethod
    def from_env(cls):
        rank = int(os.getenv('RANK', 0))
        world_size = int(os.getenv('WORLD_SIZE', 1))
        return cls(
            rank=rank,
            world_size=world_size,
            initial_rank=rank,
            initial_world_size=world_size,
        )

    def set_distributed_vars(self):
        os.environ['RANK'] = str(self.active_rank)
        os.environ['WORLD_SIZE'] = str(self.active_world_size)

    def advance(self, store=None, world_size=None, timeout=None):
        """
        Advance to the next iteration.

        If use_global_iteration is True and a store is provided, this will
        use decentralized consensus to coordinate the advancement of the global
        iteration counter. Otherwise, it will increment the local iteration counter.

        Args:
            store: Optional store instance for global iteration management
            world_size: Total number of ranks (required for global coordination)
            timeout: Optional timeout for coordination
        """
        if self.use_global_iteration and store is not None:
            # Use decentralized consensus for global iteration advancement
            import datetime

            if timeout is None:
                timeout = datetime.timedelta(seconds=30)
            if world_size is None:
                world_size = self.world_size

            self.iteration = store.coordinate_global_iteration_advance_decentralized(
                world_size, timeout
            )
        else:
            # Use local iteration counter (backward compatibility)
            self.iteration += 1

        self.fn_exception = None

    def sync_with_global_iteration(self, store, timeout=None):
        """
        Synchronize the local iteration with the global iteration counter.

        This method ensures that the local iteration matches the global
        iteration number stored in the distributed store. It also handles
        rank restarts by detecting if this rank is restarting and catching up.

        Args:
            store: Store instance for global iteration management
            timeout: Optional timeout for synchronization
        """
        if self.use_global_iteration:
            import datetime

            if timeout is None:
                timeout = datetime.timedelta(seconds=30)

            # Handle potential restart and catch up with current iteration
            self.iteration = store.handle_rank_restart(self.world_size, timeout)

    def freeze(self):
        frozen = FrozenState(**dataclasses.asdict(self))
        return frozen

    def copy_from(self, other, fields=None):
        for field in dataclasses.fields(self):
            if fields is None or field.name in fields:
                setattr(self, field.name, getattr(other, field.name))


def freeze_dataclass(cls):
    fields = [(f.name, f.type, f) for f in dataclasses.fields(cls)]
    FrozenClass = dataclasses.make_dataclass(
        cls_name=f'Frozen{cls.__name__}', fields=fields, frozen=True
    )
    return FrozenClass


FrozenState = freeze_dataclass(State)
FrozenState.__doc__ = r'''
    :py:class:`inprocess.FrozenState` is identical to
    :py:class:`inprocess.State`, except all fields are read-only.
'''
