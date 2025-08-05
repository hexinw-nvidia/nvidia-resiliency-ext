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

import datetime
import functools
import inspect
import logging
import os

# Issue: [B403:blacklist] Consider possible security implications associated with pickle module.
# Severity: Low   Confidence: High
# CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
# More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_imports.html#b403-import-pickle
import pickle  # nosec
import sys
import time
from collections.abc import Iterable
from typing import Optional

import torch

from . import exception
from .attribution import InterruptionRecord
from .state import Mode
from .utils import log_exc


class BarrierError(exception.RestartError):
    pass


class BarrierTimeout(BarrierError):
    pass


class BarrierOverflow(BarrierError):
    pass


class StoreMixin:
    ANY_RANK_INTERRUPTED = 'any_rank_interrupted'
    ANY_RANK_COMPLETED = 'any_rank_completed'

    INTERRUPTION_RECORDS = 'interruption_records'
    INTERRUPTION_RECORDS_LOCK = 'interruption_records_lock'

    TERMINATED_RANKS = 'terminated_ranks'
    INITIAL_RANK = 'initial_rank_{rank}'
    ACTIVE_RANK = 'active_rank_{rank}'
    HEARTBEAT = 'heartbeat_{rank}'

    STATE = 'state_{rank}'
    KEY = 'key_{rank}'

    BARRIER_PREFIX = 'inprocess_barrier_prefix'
    STORE_PREFIX = '_inprocess_{iteration}'

    # Global iteration management keys
    GLOBAL_ITERATION_COUNTER = 'global_iteration_counter'
    GLOBAL_ITERATION_LOCK = 'global_iteration_lock'
    GLOBAL_ITERATION_READY = 'global_iteration_ready'
    GLOBAL_ITERATION_BARRIER = 'global_iteration_barrier'

    INITIAL_BARRIER = 'initial_barrier'
    COMPLETION_BARRIER = 'completion_barrier'
    ITERATION_BARRIER = 'iteration_barrier'
    TERMINATION_BARRIER = 'termination_barrier'

    @property
    def critical_ranks(self):
        return ()

    def get_global_iteration(self) -> int:
        """
        Get the current global iteration number from the store.

        This method retrieves the globally unique iteration counter that is
        shared across all ranks and persists across process restarts.

        Returns:
            int: The current global iteration number
        """
        try:
            # Try to get the current global iteration
            iteration_str = self.get(self.GLOBAL_ITERATION_COUNTER)
            return int(iteration_str)
        except (KeyError, ValueError):
            # If the key doesn't exist or is invalid, start from 0
            return 0

    def increment_global_iteration(self) -> int:
        """
        Atomically increment the global iteration counter.

        This method ensures that only one rank can increment the counter at a time,
        providing globally unique iteration numbers across all ranks and restarts.

        Returns:
            int: The new global iteration number
        """
        # Use atomic increment to ensure thread/process safety
        new_iteration = self.add(self.GLOBAL_ITERATION_COUNTER, 1)
        return new_iteration

    def set_global_iteration(self, iteration: int) -> None:
        """
        Set the global iteration counter to a specific value.

        This is useful for initialization or when you need to reset the counter
        to a known state.

        Args:
            iteration: The iteration number to set
        """
        self.set(self.GLOBAL_ITERATION_COUNTER, str(iteration))

    def get_or_create_global_iteration(self) -> int:
        """
        Get the current global iteration or create it if it doesn't exist.

        This method is useful for initialization to ensure the global counter
        exists and starts from a known state.

        Returns:
            int: The current global iteration number
        """
        try:
            return self.get_global_iteration()
        except KeyError:
            # Initialize to 0 if it doesn't exist
            self.set_global_iteration(0)
            return 0

    def wait_for_global_iteration_ready(self, timeout: datetime.timedelta) -> int:
        """
        Wait for the global iteration to be ready for all ranks.

        This method waits for the global iteration barrier to be satisfied,
        ensuring all ranks are synchronized before proceeding.

        Args:
            timeout: Maximum time to wait for readiness

        Returns:
            int: The current global iteration number
        """
        try:
            # Wait for the global iteration to be ready
            self.wait([self.GLOBAL_ITERATION_READY], timeout)
            return self.get_global_iteration()
        except Exception:
            # Fallback: return current iteration
            return self.get_global_iteration()

    def signal_global_iteration_ready(self, iteration: int) -> None:
        """
        Signal that the global iteration is ready for all ranks.

        This method is called by the consensus winner to signal that
        all ranks can proceed with the current iteration.

        Args:
            iteration: The iteration number that is ready
        """
        self.set(self.GLOBAL_ITERATION_READY, str(iteration))

    def clear_global_iteration_ready(self) -> None:
        """
        Clear the global iteration ready signal.

        This is called after all ranks have acknowledged the iteration
        to prepare for the next iteration.
        """
        try:
            self.delete_key(self.GLOBAL_ITERATION_READY)
        except KeyError:
            pass

    def reset_initial_barrier(self) -> None:
        """
        Reset the initial barrier state after all ranks have completed it.

        This method clears all keys related to the initial barrier to ensure
        that restarted ranks don't get stuck on old barrier state from
        previous startup attempts.
        """
        barrier_key = f'{self.BARRIER_PREFIX}:barrier:{self.INITIAL_BARRIER}'
        last_worker_arrived_key = f'{barrier_key}:last_worker_arrived'
        arrived_key = f'{barrier_key}:arrived'

        # Clear all initial barrier related keys
        try:
            self.delete_key(barrier_key)
        except KeyError:
            pass
        try:
            self.delete_key(last_worker_arrived_key)
        except KeyError:
            pass
        try:
            self.delete_key(arrived_key)
        except KeyError:
            pass

        logging.getLogger(__name__).info("Initial barrier state reset completed")

    def coordinate_global_iteration_advance_decentralized(
        self, world_size: int, timeout: datetime.timedelta
    ) -> int:
        """
        Coordinate the advancement of global iteration using decentralized consensus.

        This method implements a consensus protocol where all ranks participate equally:
        1. All active ranks signal readiness for next iteration
        2. When all active ranks are ready, any rank can propose advancement
        3. Consensus is reached through atomic operations
        4. No single point of failure or coordinator required

        Args:
            world_size: Total number of ranks in the distributed system
            timeout: Maximum time to wait for consensus

        Returns:
            int: The new global iteration number
        """
        rank = int(os.environ.get('RANK', 0))

        # Step 1: Wait for current iteration to be ready
        current_iteration = self.wait_for_global_iteration_ready(timeout)

        # Step 2: Signal readiness for next iteration
        ready_key = f'{self.GLOBAL_ITERATION_BARRIER}:ready'
        arrived_key = f'{self.GLOBAL_ITERATION_BARRIER}:arrived'

        # Signal that we're ready for next iteration
        self.append(arrived_key, f'{rank},')

        # Step 3: Get list of active ranks (ranks that haven't crashed)
        active_ranks = self.get_active_ranks_for_coordination(world_size, timeout)

        # Step 4: Wait for all active ranks to signal readiness
        arrived_ranks = set([int(r) for r in self.get_packed(arrived_key, ',') if r.strip()])
        active_arrived = arrived_ranks.intersection(active_ranks)
        arrived_count = len(active_arrived)

        logging.getLogger(__name__).info(
            f"Rank {rank}: Active ranks: {active_ranks}, "
            f"Arrived ranks: {arrived_ranks}, "
            f"Active arrived: {active_arrived}, "
            f"Count: {arrived_count}/{len(active_ranks)}"
        )

        # Step 5: Consensus-based advancement
        if arrived_count >= len(active_ranks):
            # All active ranks are ready, attempt to advance
            # Use atomic compare-and-swap to ensure only one rank advances
            consensus_key = f'{self.GLOBAL_ITERATION_BARRIER}:consensus'

            try:
                # Try to set consensus flag atomically
                # Only one rank will succeed in setting this flag
                self.set(consensus_key, str(rank))

                # Check if we won the consensus (we set the flag)
                consensus_winner = int(self.get(consensus_key))

                if consensus_winner == rank:
                    # We won the consensus, advance the iteration
                    logging.getLogger(__name__).info(
                        f"Rank {rank}: Won consensus, advancing iteration"
                    )

                    new_iteration = self.increment_global_iteration()

                    # Record the new iteration for all active ranks
                    for active_rank in active_ranks:
                        self.record_iteration_for_rank(active_rank, new_iteration)

                    # Clear the ready signal for current iteration
                    self.clear_global_iteration_ready()

                    # Signal that new iteration is ready
                    self.signal_global_iteration_ready(new_iteration)

                    # Clear the barrier state
                    try:
                        self.delete_key(ready_key)
                    except KeyError:
                        pass
                    try:
                        self.delete_key(arrived_key)
                    except KeyError:
                        pass
                    try:
                        self.delete_key(consensus_key)
                    except KeyError:
                        pass

                    return new_iteration
                else:
                    # Another rank won the consensus, wait for the new iteration
                    logging.getLogger(__name__).info(
                        f"Rank {rank}: Lost consensus to Rank {consensus_winner}, waiting for new iteration"
                    )

                    try:
                        self.wait([self.GLOBAL_ITERATION_READY], timeout)
                        new_iteration = self.get_global_iteration()

                        # Record the new iteration for this rank
                        self.record_iteration_for_rank(rank, new_iteration)

                        return new_iteration
                    except Exception:
                        return current_iteration

            except Exception:
                # Consensus failed, wait for new iteration
                try:
                    self.wait([self.GLOBAL_ITERATION_READY], timeout)
                    return self.get_global_iteration()
                except Exception:
                    return current_iteration
        else:
            # Not all active ranks ready yet, wait with timeout
            try:
                self.wait([ready_key], timeout)
                return self.get_global_iteration()
            except Exception:
                # Timeout occurred, check if we should proceed anyway
                if arrived_count >= max(1, len(active_ranks) // 2):  # At least half of active ranks
                    logging.getLogger(__name__).warning(
                        f"Rank {rank}: Timeout reached, proceeding with {arrived_count}/{len(active_ranks)} active ranks"
                    )

                    # Use consensus for partial quorum as well
                    consensus_key = f'{self.GLOBAL_ITERATION_BARRIER}:consensus'

                    try:
                        self.set(consensus_key, str(rank))
                        consensus_winner = int(self.get(consensus_key))

                        if consensus_winner == rank:
                            new_iteration = self.increment_global_iteration()

                            # Record the new iteration for arrived ranks
                            for arrived_rank in active_arrived:
                                self.record_iteration_for_rank(arrived_rank, new_iteration)

                            self.clear_global_iteration_ready()
                            self.signal_global_iteration_ready(new_iteration)
                            return new_iteration
                        else:
                            try:
                                self.wait([self.GLOBAL_ITERATION_READY], timeout)
                                return self.get_global_iteration()
                            except Exception:
                                return current_iteration
                    except Exception:
                        return current_iteration
                else:
                    return current_iteration

    def get_active_ranks_for_coordination(
        self, world_size: int, timeout: datetime.timedelta
    ) -> set[int]:
        """
        Get the list of active ranks for coordination, excluding crashed ranks.

        This method detects which ranks are still active by checking heartbeats
        and excludes crashed ranks from the coordination protocol.

        Args:
            world_size: Total number of ranks in the distributed system
            timeout: Maximum time to wait for detection

        Returns:
            set[int]: Set of active rank numbers
        """
        active_ranks = set()
        current_time = time.time()

        # Check heartbeats for all ranks
        for rank in range(world_size):
            try:
                heartbeat_time = self.get_heartbeat(rank)
                # Consider rank active if heartbeat is within last 30 seconds
                if current_time - heartbeat_time < 30:
                    active_ranks.add(rank)
                else:
                    logging.getLogger(__name__).warning(
                        f"Rank {rank} appears inactive (last heartbeat: {heartbeat_time})"
                    )
            except Exception:
                # Rank has no heartbeat, consider it inactive
                logging.getLogger(__name__).warning(
                    f"Rank {rank} has no heartbeat, considering inactive"
                )

        # Ensure we have at least one active rank
        if not active_ranks:
            active_ranks.add(0)  # Fallback to rank 0

        logging.getLogger(__name__).info(f"Active ranks for coordination: {active_ranks}")
        return active_ranks

    def detect_crashed_ranks(self, world_size: int) -> set[int]:
        """
        Detect ranks that have crashed or are unresponsive.

        Args:
            world_size: Total number of ranks in the distributed system

        Returns:
            set[int]: Set of crashed/unresponsive rank numbers
        """
        all_ranks = set(range(world_size))
        active_ranks = self.get_active_ranks_for_coordination(
            world_size, datetime.timedelta(seconds=10)
        )
        crashed_ranks = all_ranks - active_ranks

        if crashed_ranks:
            logging.getLogger(__name__).warning(f"Detected crashed ranks: {crashed_ranks}")

        return crashed_ranks

    def synchronize_global_iteration(self, world_size: int, timeout: datetime.timedelta) -> int:
        """
        Synchronize the global iteration across all ranks.

        This method ensures all ranks have the same global iteration number
        using decentralized consensus.

        Args:
            world_size: Total number of ranks in the distributed system
            timeout: Maximum time to wait for synchronization

        Returns:
            int: The synchronized global iteration number
        """
        # Use decentralized consensus for synchronization
        return self.coordinate_global_iteration_advance_decentralized(world_size, timeout)

    def get_packed(self, key: str, sep: str):
        return self.get(key).decode().rstrip(sep).split(sep)

    def set_active_rank(self, rank, active):
        match active:
            case Mode.ACTIVE:
                active_str = '1'
            case Mode.INACTIVE:
                active_str = ''
            case _:
                raise RuntimeError
        self.set(self.ACTIVE_RANK.format(rank=rank), active_str)

    def get_all_active_ranks(self, world_size):
        return [
            bool(active)
            for active in self.multi_get(
                [self.ACTIVE_RANK.format(rank=rank) for rank in range(world_size)]
            )
        ]

    def set_initial_rank(self, rank, initial_rank):
        self.set(self.INITIAL_RANK.format(rank=rank), str(initial_rank))

    def get_initial_ranks(self, ranks: Iterable[int]) -> list[int]:
        return self.multi_get([self.INITIAL_RANK.format(rank=rank) for rank in ranks])

    def send_heartbeat(self, rank: int):
        self.set(self.HEARTBEAT.format(rank=rank), str(time.time()))

    def send_state(self, state, rank: int):
        self.set(self.STATE.format(rank=rank), pickle.dumps(state))

    def send_key(self, key, rank: int):
        self.set(self.KEY.format(rank=rank), pickle.dumps(key))

    def get_states(self, ranks):
        states = [
            # Issue: [B301:blacklist] Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue.
            # Severity: Medium   Confidence: High
            # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
            # More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_calls.html#b301-pickle
            pickle.loads(state)  # nosec
            for state in self.multi_get([self.STATE.format(rank=rank) for rank in ranks])
        ]
        return states

    def get_keys(self, ranks):
        keys = [
            # Issue: [B301:blacklist] Pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue.
            # Severity: Medium   Confidence: High
            # CWE: CWE-502 (https://cwe.mitre.org/data/definitions/502.html)
            # More Info: https://bandit.readthedocs.io/en/1.8.3/blacklists/blacklist_calls.html#b301-pickle
            pickle.loads(key)  # nosec
            for key in self.multi_get([self.KEY.format(rank=rank) for rank in ranks])
        ]
        return keys

    def get_heartbeat(self, rank: int) -> float:
        return float(self.get(self.HEARTBEAT.format(rank=rank)))

    def get_all_heartbeats(self, world_size: int) -> list[float]:
        return [
            float(heartbeat)
            for heartbeat in self.multi_get(
                [self.HEARTBEAT.format(rank=rank) for rank in range(world_size)]
            )
        ]

    def record_interrupted(self, records: Optional[Iterable[InterruptionRecord]] = None):
        if records is not None:
            self.append(self.INTERRUPTION_RECORDS, '')

            records_are_locked = bool(self.add(self.INTERRUPTION_RECORDS_LOCK, 0))
            if not records_are_locked:
                msg = ';'.join(str(r) for r in records)
                self.append(self.INTERRUPTION_RECORDS, f'{msg};')

        self.set(self.ANY_RANK_INTERRUPTED, '')

    def lock_interruption_records(self):
        self.add(self.INTERRUPTION_RECORDS_LOCK, 1)

    def get_interruption_records(self) -> list[InterruptionRecord]:
        self.append(self.INTERRUPTION_RECORDS, '')
        records = [
            InterruptionRecord.from_str(record)
            for record in self.get_packed(self.INTERRUPTION_RECORDS, ';')
            if record.strip()
        ]
        return records

    def wait_for_interrupted(self, timeout: datetime.timedelta):
        self.wait([self.ANY_RANK_INTERRUPTED], timeout)

    def wait_for_completed(self, timeout: datetime.timedelta):
        self.wait([self.ANY_RANK_COMPLETED], timeout)

    def record_completed(self):
        self.set(self.ANY_RANK_COMPLETED, '')

    def record_terminated_ranks(self, ranks: Iterable[int]):
        ranks_str = ','.join(str(r) for r in ranks)
        if ranks_str:
            self.append(self.TERMINATED_RANKS, f'{ranks_str},')

    def get_terminated_ranks(self) -> set[int]:
        self.append(self.TERMINATED_RANKS, '')
        terminated_ranks = set(
            [int(r) for r in self.get_packed(self.TERMINATED_RANKS, ',') if r.strip()]
        )
        return terminated_ranks

    def barrier(
        self,
        ranks: Iterable[int],
        group_name: str,
        rendezvous_count: int,
        timeout: datetime.timedelta,
        timeout_chunk: Optional[datetime.timedelta] = None,
    ):
        log = logging.getLogger(__name__)
        cn = inspect.currentframe().f_code.co_name
        log.debug(f'{ranks=} enter {group_name=} {cn} {rendezvous_count=}')

        store_key = f'{self.BARRIER_PREFIX}:{cn}:{group_name}'
        last_worker_arrived_key = f'{store_key}:last_worker_arrived'
        arrived_key = f'{store_key}:arrived'

        # Debug: Check the value before atomic increment
        try:
            # Wait for the key to exist with a short timeout
            self.wait([store_key], datetime.timedelta(seconds=1))
            before_value = self.get(store_key)
            sys.stderr.write(
                f'Rank {list(ranks)[0]}: Before atomic increment, {store_key} = {before_value}\n'
            )
        except Exception as e:
            sys.stderr.write(
                f'Rank {list(ranks)[0]}: Before atomic increment, {store_key} does not exist: {e}\n'
            )

        # Debug: Perform atomic increment and capture the result
        increment_amount = len(set(ranks))
        arrived_count = self.add(store_key, increment_amount)
        sys.stderr.write(
            f'Rank {list(ranks)[0]}: Atomic increment by {increment_amount}, new count = {arrived_count}\n'
        )

        # Debug: Check the value after atomic increment
        try:
            # Wait for the key to exist with a short timeout
            self.wait([store_key], datetime.timedelta(seconds=1))
            after_value = self.get(store_key)
            sys.stderr.write(
                f'Rank {list(ranks)[0]}: After atomic increment, {store_key} = {after_value}\n'
            )
        except Exception as e:
            sys.stderr.write(
                f'Rank {list(ranks)[0]}: After atomic increment, {store_key} error: {e}\n'
            )

        ranks_str = ','.join(str(r) for r in ranks)
        self.append(arrived_key, f'{ranks_str},')

        if arrived_count > rendezvous_count:
            arrived_ranks = sorted([int(r) for r in self.get_packed(arrived_key, ',') if r.strip()])
            raise BarrierOverflow(f'{ranks=} {rendezvous_count=} {group_name=} {arrived_ranks=}')

        if arrived_count == rendezvous_count:
            self.set(last_worker_arrived_key, '1')

        if timeout_chunk is None:
            timeout_chunk = timeout
        else:
            timeout_chunk = min(timeout_chunk, timeout)

        if timeout and timeout_chunk:
            start = time.monotonic()
            while True:
                try:
                    self.wait([last_worker_arrived_key], timeout_chunk)
                    break
                except torch.distributed.DistStoreError as ex:
                    if datetime.timedelta(seconds=(time.monotonic() - start)) > timeout:
                        raise BarrierTimeout(
                            f'{ranks=} {rendezvous_count=} {group_name=} ' f'{timeout=}'
                        ) from ex
                    time.sleep(sys.getswitchinterval())

        log.debug(f'{ranks=} exits {group_name=} {cn} {rendezvous_count=}')

    def is_rank_at_reentrant_barrier(
        self,
        rank: int,
        group_name: str,
    ):
        log = logging.getLogger(__name__)
        barrier_name = self.reentrant_barrier.__name__
        store_key = f'{self.BARRIER_PREFIX}:{barrier_name}:{group_name}'
        arrived_key = f'{store_key}:arrived'
        self.append(arrived_key, '')
        arrived_ranks = set([int(r) for r in self.get_packed(arrived_key, ',') if r.strip()])
        log.debug(f'{rank=} {arrived_ranks=}')
        arrived = rank in arrived_ranks
        if arrived:
            log.debug(f'{rank=} already arrived {group_name=}')
        return arrived

    def reentrant_barrier(
        self,
        ranks: Iterable[int],
        group_name: str,
        rendezvous_count: int,
        timeout: datetime.timedelta,
        timeout_chunk: Optional[datetime.timedelta] = None,
    ):
        log = logging.getLogger(__name__)
        cn = inspect.currentframe().f_code.co_name
        log.debug(f'{ranks=} enter {group_name=} {cn} {rendezvous_count=}')

        store_key = f'{self.BARRIER_PREFIX}:{cn}:{group_name}'
        last_worker_arrived_key = f'{store_key}:last_worker_arrived'
        arrived_key = f'{store_key}:arrived'

        ranks_str = ','.join(str(r) for r in ranks)
        self.append(arrived_key, f'{ranks_str},')

        arrived_ranks = set([int(r) for r in self.get_packed(arrived_key, ',') if r.strip()])
        arrived_count = len(arrived_ranks)

        if arrived_count > rendezvous_count:
            arrived_ranks = sorted(list(arrived_ranks))
            raise BarrierOverflow(f'{ranks=} {rendezvous_count=} {group_name=} {arrived_ranks=}')

        if arrived_count == rendezvous_count:
            self.set(last_worker_arrived_key, '1')

        if timeout_chunk is None:
            timeout_chunk = timeout
        else:
            timeout_chunk = min(timeout_chunk, timeout)

        if timeout and timeout_chunk:
            start = time.monotonic()
            while True:
                try:
                    self.wait([last_worker_arrived_key], timeout_chunk)
                    break
                except torch.distributed.DistStoreError as ex:
                    if datetime.timedelta(seconds=(time.monotonic() - start)) > timeout:
                        raise BarrierTimeout(
                            f'{ranks=} {rendezvous_count=} {group_name=} ' f'{timeout=}'
                        ) from ex
                    time.sleep(sys.getswitchinterval())

        log.debug(f'{ranks=} exits {group_name=} {cn} {rendezvous_count=}')

    initial_barrier = functools.partialmethod(
        barrier,
        group_name=INITIAL_BARRIER,
    )
    completion_barrier = functools.partialmethod(
        barrier,
        group_name=COMPLETION_BARRIER,
    )
    iteration_barrier = functools.partialmethod(
        reentrant_barrier,
        group_name=ITERATION_BARRIER,
    )
    termination_barrier = functools.partialmethod(
        reentrant_barrier,
        group_name=TERMINATION_BARRIER,
    )

    def catch_up_restarting_rank(self, timeout: datetime.timedelta) -> int:
        """
        Handle a restarting rank that needs to catch up with the current global iteration.

        This method is called by a rank that has restarted and needs to:
        1. Get the current global iteration number
        2. Signal that it's ready to participate in coordination
        3. Wait for the next iteration to be ready
        4. Handle the case where iteration advances while restarting

        Args:
            timeout: Maximum time to wait for catch-up

        Returns:
            int: The current global iteration number
        """
        rank = int(os.environ.get('RANK', 0))

        # Step 1: Get the current global iteration
        current_iteration = self.get_global_iteration()

        # Step 2: Signal that this rank is back online
        self.send_heartbeat(rank)

        # Step 3: Check if we need to wait for current iteration or next iteration
        try:
            # Try to wait for the current iteration to be ready
            self.wait([self.GLOBAL_ITERATION_READY], timeout)
            ready_iteration = self.get_global_iteration()

            # If the ready iteration is different from what we started with,
            # it means the iteration advanced while we were restarting
            if ready_iteration != current_iteration:
                logging.getLogger(__name__).info(
                    f"Rank {rank}: Iteration advanced from {current_iteration} to {ready_iteration} during restart"
                )
                return ready_iteration
            else:
                return current_iteration

        except Exception:
            # If no ready signal, check if iteration has advanced
            latest_iteration = self.get_global_iteration()
            if latest_iteration != current_iteration:
                logging.getLogger(__name__).info(
                    f"Rank {rank}: Iteration advanced from {current_iteration} to {latest_iteration} during restart"
                )
                return latest_iteration
            else:
                # Return current iteration if no advancement occurred
                return current_iteration

    def handle_rank_restart(self, world_size: int, timeout: datetime.timedelta) -> int:
        """
        Handle a rank that has restarted and needs to rejoin the coordination.

        This method implements the restart protocol:
        1. Detect if this rank is restarting
        2. Catch up with current global iteration
        3. Handle race conditions where iteration advances during restart
        4. Signal readiness to participate in coordination
        5. Wait for next iteration

        Args:
            world_size: Total number of ranks in the distributed system
            timeout: Maximum time to wait for restart handling

        Returns:
            int: The current global iteration number
        """
        rank = int(os.environ.get('RANK', 0))

        # Check if this rank appears to be restarting (no recent heartbeat)
        try:
            last_heartbeat = self.get_heartbeat(rank)
            current_time = time.time()
            is_restarting = (current_time - last_heartbeat) > 30
        except Exception:
            # No heartbeat found, definitely restarting
            is_restarting = True

        if is_restarting:
            logging.getLogger(__name__).info(
                f"Rank {rank}: Detected restart, catching up with global iteration"
            )

            # Get the iteration we were at before restart
            try:
                previous_iteration = int(self.get(f'rank_{rank}_last_iteration'))
            except Exception:
                # No previous iteration recorded, use current
                previous_iteration = self.get_global_iteration()

            # Catch up with current iteration (handles race conditions)
            current_iteration = self.catch_up_restarting_rank(timeout)

            # Record that we've caught up
            self.set(f'rank_{rank}_last_iteration', str(current_iteration))

            logging.getLogger(__name__).info(
                f"Rank {rank}: Restart complete. Previous: {previous_iteration}, Current: {current_iteration}"
            )

            return current_iteration
        else:
            # Normal operation, just get current iteration
            current_iteration = self.get_global_iteration()
            # Record current iteration for future restart detection
            self.set(f'rank_{rank}_last_iteration', str(current_iteration))
            return current_iteration

    def record_iteration_for_rank(self, rank: int, iteration: int) -> None:
        """
        Record the current iteration for a specific rank.

        This is used to track what iteration each rank was at,
        which helps with restart recovery and race condition handling.

        Args:
            rank: The rank number
            iteration: The iteration number to record
        """
        self.set(f'rank_{rank}_last_iteration', str(iteration))

    def get_iteration_for_rank(self, rank: int) -> int:
        """
        Get the last recorded iteration for a specific rank.

        Args:
            rank: The rank number

        Returns:
            int: The last recorded iteration for the rank
        """
        try:
            return int(self.get(f'rank_{rank}_last_iteration'))
        except Exception:
            # If no record exists, return current global iteration
            return self.get_global_iteration()


class TCPStore(torch.distributed.TCPStore, StoreMixin):
    TCP_STORE_HOST_RANK = 0

    def __init__(
        self,
        host_name: Optional[str] = None,
        port: Optional[int] = None,
        world_size: Optional[int] = None,
        timeout: datetime.timedelta = datetime.timedelta(seconds=300),
        wait_for_workers: bool = True,
        multi_tenant: bool = False,
        use_libuv: bool = True,
        tcp_store_host_rank: Optional[int] = None,
    ):
        log = logging.getLogger(__name__)

        if host_name is None:
            host_name = os.environ['MASTER_ADDR']
        if port is None:
            port = int(os.environ['MASTER_PORT'])
        if world_size is None:
            world_size = int(os.environ['WORLD_SIZE'])

        rank = int(os.environ['RANK'])

        # Save the host rank for later use
        self.tcp_store_host_rank = tcp_store_host_rank or self.TCP_STORE_HOST_RANK

        kwargs = {
            'host_name': host_name,
            'port': port,
            'world_size': world_size,
            'timeout': timeout,
            'wait_for_workers': wait_for_workers,
            'multi_tenant': multi_tenant,
            'use_libuv': use_libuv,
        }

        if rank == self.tcp_store_host_rank:
            try:
                super().__init__(is_master=True, **kwargs)
                log.debug(f'{rank=} hosting {type(self).__name__}({kwargs})')
            except Exception as store_ex:
                log.debug(log_exc(rank, store_ex, 'store_ex'))
                super().__init__(is_master=False, **kwargs)
        else:
            super().__init__(is_master=False, **kwargs)

        # Log successful connection
        if rank == 0:
            log.info(f'Rank {rank}: Successfully connected to TCPStore at {host_name}:{port}')

    @property
    def critical_ranks(self):
        # If TCP_STORE_HOST_RANK is -1, there are no critical ranks
        # (all ranks are clients to external service)
        if self.tcp_store_host_rank == -1:
            return ()
        return (self.tcp_store_host_rank,)


class PrefixStore(torch.distributed.PrefixStore, StoreMixin):
    def __init__(self, iteration, store):
        prefix = self.STORE_PREFIX.format(iteration=iteration)
        self.base_store = store
        super().__init__(prefix, store)


class FileStore(torch.distributed.FileStore, StoreMixin):
    pass
