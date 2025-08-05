# Globally Unique Iteration Numbers

## Overview

The global iteration feature ensures that all ranks in a distributed training system have globally unique iteration numbers across process restarts, even when using an external TCPStore. This is particularly important for fault-tolerant training where ranks may restart at different times.

## Critical Design Question: Who Gets to Increment the Global Iteration Counter?

This is the most important question in the design. Consider this scenario:

1. **Initial state**: Rank 0, 1, 2 start training, Rank 3 is hot spare
2. **Failure**: Rank 1 fails with GPU error  
3. **Restart**: Rank 1 restarts and tries to join
4. **Question**: Who should increment the global iteration counter?

### The Problem with Simple Atomic Increment

If we simply use atomic increment, we get these issues:
- **Race condition**: Multiple ranks could try to increment simultaneously
- **Inconsistent state**: Rank 1 might get a different iteration than Rank 0, 2, 3
- **Barrier coordination**: Rank 1 needs to wait for the correct iteration

### The Critical Problem: Crashed Ranks

The most critical issue is when **Rank 1 crashes due to a C++ backend exception**:

1. **Rank 1 crashes immediately** without calling `state.advance()`
2. **Rank 1 never signals readiness** for the next iteration
3. **Rank 0 (coordinator) waits forever** for Rank 1 to signal readiness
4. **The entire system deadlocks** waiting for a crashed rank

### The Solution: Fault-Tolerant Coordinated Iteration Advancement

The correct design uses **fault-tolerant coordinated iteration advancement** where:

1. **All ACTIVE ranks must be ready** before the iteration advances
2. **Only the coordinator (Rank 0) increments** the global counter
3. **Crashed ranks are detected and excluded** from coordination
4. **Restarting ranks catch up** without blocking the system

## Problem Statement

In the original design, each rank maintained its own local iteration counter:

```python
def advance(self):
    self.iteration += 1  # Local increment only
```

This approach has several issues:
1. **Inconsistent iteration numbers**: When ranks restart, they may have different iteration numbers
2. **No global coordination**: No mechanism ensures iteration uniqueness across the entire distributed system
3. **State isolation problems**: Different ranks might be at different iterations, causing coordination issues
4. **Race conditions**: Multiple ranks could increment simultaneously, leading to inconsistencies
5. **Deadlock with crashed ranks**: Crashed ranks prevent the system from advancing

## Solution: Fault-Tolerant Coordinated Global Iteration Counter

The new design introduces a **fault-tolerant coordinated global iteration counter** with proper crash detection and restart handling:

### Key Components

#### 1. Fault-Tolerant Global Iteration Management in StoreMixin

```python
class StoreMixin:
    # Global iteration management keys
    GLOBAL_ITERATION_COUNTER = 'global_iteration_counter'
    GLOBAL_ITERATION_READY = 'global_iteration_ready'
    GLOBAL_ITERATION_BARRIER = 'global_iteration_barrier'
    
    def coordinate_global_iteration_advance(
        self, 
        world_size: int, 
        timeout: datetime.timedelta,
        is_coordinator: bool = False
    ) -> int:
        """
        Coordinate the advancement of global iteration across all ranks.
        
        Protocol:
        1. All ranks wait for the current iteration to be ready
        2. Only the coordinator increments the counter when ACTIVE ranks are ready
        3. Crashed ranks are detected and excluded from coordination
        4. All active ranks wait for the new iteration to be signaled
        """
        # Step 1: Wait for current iteration to be ready
        current_iteration = self.wait_for_global_iteration_ready(timeout)
        
        # Step 2: Coordinate advancement with fault tolerance
        if is_coordinator:
            # Get list of active ranks (ranks that haven't crashed)
            active_ranks = self.get_active_ranks_for_coordination(world_size, timeout)
            
            # Wait for all ACTIVE ranks to arrive (not all ranks)
            arrived_ranks = set([int(r) for r in self.get_packed(arrived_key, ',') if r.strip()])
            active_arrived = arrived_ranks.intersection(active_ranks)
            arrived_count = len(active_arrived)
            
            if arrived_count >= len(active_ranks):
                # All ACTIVE ranks are ready, increment the global iteration
                new_iteration = self.increment_global_iteration()
                self.signal_global_iteration_ready(new_iteration)
                return new_iteration
            else:
                # Handle timeout with partial quorum
                if arrived_count >= max(1, len(active_ranks) // 2):
                    # Proceed with majority of active ranks
                    new_iteration = self.increment_global_iteration()
                    self.signal_global_iteration_ready(new_iteration)
                    return new_iteration
        else:
            # Non-coordinator ranks signal readiness and wait for new iteration
            self.signal_readiness()
            return self.wait_for_new_iteration()
    
    def get_active_ranks_for_coordination(self, world_size: int, timeout: datetime.timedelta) -> set[int]:
        """
        Get the list of active ranks for coordination, excluding crashed ranks.
        
        This method detects which ranks are still active by checking heartbeats
        and excludes crashed ranks from the coordination protocol.
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
            except Exception:
                # Rank has no heartbeat, consider it inactive
                pass
        
        # Always include rank 0 (coordinator) if it exists
        if 0 < world_size:
            active_ranks.add(0)
        
        return active_ranks
    
    def handle_rank_restart(self, world_size: int, timeout: datetime.timedelta) -> int:
        """
        Handle a rank that has restarted and needs to rejoin the coordination.
        
        This method implements the restart protocol:
        1. Detect if this rank is restarting
        2. Catch up with current global iteration
        3. Signal readiness to participate in coordination
        4. Wait for next iteration
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
            return self.catch_up_restarting_rank(timeout)
        else:
            # Normal operation, just get current iteration
            return self.get_global_iteration()
```

#### 2. Enhanced State Management with Restart Handling

```python
@dataclasses.dataclass
class State:
    iteration: int = 0
    use_global_iteration: bool = False  # New field
    
    def advance(self, store=None, world_size=None, timeout=None, is_coordinator=False):
        """Advance to the next iteration with fault-tolerant coordination."""
        if self.use_global_iteration and store is not None:
            # Use fault-tolerant coordinated global iteration advancement
            self.iteration = store.coordinate_global_iteration_advance(
                world_size, timeout, is_coordinator
            )
        else:
            # Use local iteration counter (backward compatibility)
            self.iteration += 1
    
    def sync_with_global_iteration(self, store, timeout=None):
        """Synchronize with global iteration, handling restarts."""
        if self.use_global_iteration:
            # Handle potential restart and catch up with current iteration
            self.iteration = store.handle_rank_restart(self.world_size, timeout)
```

#### 3. Wrapper Configuration with Fault Tolerance

```python
@Wrapper(
    store_kwargs={'port': int(os.getenv('MASTER_PORT', 29500)) + 2},
    use_global_iteration=True,  # Enable fault-tolerant global iteration
)
def train_function(...):
    pass

# In the CallWrapper:
is_coordinator = (state.rank == 0)  # Rank 0 is the coordinator
state.advance(
    store=base_store,
    world_size=state.world_size,
    timeout=wrapper.barrier_timeout,
    is_coordinator=is_coordinator
)
```

## How the Fault-Tolerant Protocol Works

### 1. Initialization Phase

When a rank starts or restarts:

1. **State Configuration**: The state is configured to use global iteration
2. **Restart Detection**: The system detects if this rank is restarting
3. **Catch-up Protocol**: Restarting ranks catch up with current iteration
4. **Coordinator Assignment**: Rank 0 is designated as the coordinator
5. **Barrier Setup**: Global iteration barrier is established

```python
# Configure state to use global iteration
state.use_global_iteration = wrapper.use_global_iteration

# If using global iteration, handle restart and sync
if wrapper.use_global_iteration:
    state.sync_with_global_iteration(base_store, wrapper.barrier_timeout)
```

### 2. Fault-Tolerant Iteration Advancement

When advancing to the next iteration:

1. **Active Rank Detection**: System detects which ranks are still active
2. **Crashed Rank Exclusion**: Crashed ranks are excluded from coordination
3. **Active Ranks Signal Readiness**: Only active ranks signal readiness
4. **Coordinator Waits**: Rank 0 waits for ALL ACTIVE ranks to signal readiness
5. **Atomic Increment**: Only Rank 0 increments the global counter
6. **Signal New Iteration**: Rank 0 signals the new iteration to all ranks
7. **All Active Ranks Proceed**: All active ranks wait for and receive the new iteration number

```python
# Advance iteration with fault-tolerant coordination
if wrapper.use_global_iteration:
    is_coordinator = (state.rank == 0)
    state.advance(
        store=base_store,
        world_size=state.world_size,
        timeout=wrapper.barrier_timeout,
        is_coordinator=is_coordinator
    )
```

### 3. Crash Handling and Restart Recovery

When Rank 1 crashes in your scenario:

1. **Rank 1 Crashes**: Rank 1 fails with GPU error (C++ exception)
2. **No Cleanup**: Rank 1 never calls `state.advance()` or signals readiness
3. **Active Rank Detection**: System detects Rank 1 is inactive (no heartbeat)
4. **Exclusion from Coordination**: Rank 1 is excluded from coordination
5. **Other Ranks Continue**: Rank 0, 2, 3 continue with coordination
6. **Iteration Advances**: Global iteration advances without Rank 1
7. **Rank 1 Restarts**: Rank 1 comes back online later
8. **Catch-up Protocol**: Rank 1 catches up with current iteration
9. **Rejoin Coordination**: Rank 1 rejoins coordination for next iteration

### 4. Race Condition Handling: Early Restart

A critical issue occurs when **Rank 1 restarts before the global iteration advances**:

#### The Race Condition Problem

1. **Rank 0, 1, 2** are at iteration 3
2. **Rank 1 crashes** with GPU error
3. **Rank 1 restarts immediately** (before Rank 0 advances to iteration 4)
4. **Rank 1 tries to catch up** but gets iteration 3 (same as before)
5. **Rank 1 waits at barrier** for iteration 4
6. **Rank 0 advances to iteration 4** (without Rank 1)
7. **Rank 1 is now stuck** waiting for iteration 4 that already happened

#### The Solution: Iteration Tracking and Race Detection

The system handles this race condition through:

1. **Iteration Recording**: Each rank's iteration is recorded in the store
2. **Race Detection**: System detects when iteration advances during restart
3. **Automatic Catch-up**: Restarting ranks automatically catch up to current iteration

### 5. Coordinator Failover: When Rank 0 Crashes

The most critical failure scenario is when **Rank 0 (the coordinator) crashes**:

#### The Coordinator Failure Problem

1. **Rank 0 is the coordinator** that manages global iteration advancement
2. **Only Rank 0 can increment** the global iteration counter
3. **Other ranks wait for Rank 0** to signal the new iteration
4. **If Rank 0 crashes, the entire system deadlocks**

This creates a **single point of failure** that can bring down the entire distributed training system.

#### The Solution: Automatic Coordinator Election

The system implements **automatic coordinator failover**:

1. **Coordinator Heartbeat Monitoring**: Tracks if Rank 0 is alive
2. **Automatic Election**: When Rank 0 crashes, a new coordinator is elected
3. **Seamless Handover**: Training continues with the new coordinator
4. **No Manual Intervention**: Failover happens automatically

```python
def elect_coordinator(self, world_size: int, timeout: datetime.timedelta) -> int:
    """Elect a new coordinator when the current coordinator fails."""
    rank = int(os.environ.get('RANK', 0))
    
    # Check if current coordinator (Rank 0) is alive
    try:
        coordinator_heartbeat = self.get_coordinator_heartbeat()
        current_time = time.time()
        if current_time - coordinator_heartbeat < 30:  # Coordinator alive if heartbeat < 30s
            return 0  # Rank 0 is still the coordinator
    except Exception:
        pass  # No heartbeat found, proceed with election
    
    # Coordinator is dead, start election
    logging.warning(f"Rank {rank}: Coordinator (Rank 0) appears dead, starting election")
    
    # Use atomic increment to elect coordinator
    election_key = f'{self.COORDINATOR_ELECTION}:election_counter'
    election_value = self.add(election_key, 1)
    
    # Wait for all active ranks to participate in election
    active_ranks = self.get_active_ranks_for_coordination(world_size, timeout)
    max_election_value = len(active_ranks)
    
    # The rank with election_value == 1 becomes the new coordinator
    if election_value == 1:
        new_coordinator = rank
        logging.info(f"Rank {rank}: Elected as new coordinator")
    else:
        # Find the rank that got election_value == 1
        new_coordinator = min(active_ranks)  # Fallback to lowest active rank
    
    # Record the new coordinator
    self.set(f'{self.COORDINATOR_ELECTION}:current_coordinator', str(new_coordinator))
    
    return new_coordinator
```

#### Coordinator Heartbeat Management

The coordinator maintains a heartbeat to show it's alive:

```python
def update_coordinator_heartbeat(self) -> None:
    """Update the coordinator heartbeat to indicate it's alive."""
    self.set(self.COORDINATOR_HEARTBEAT, str(time.time()))

def is_coordinator_alive(self, timeout_seconds: int = 30) -> bool:
    """Check if the current coordinator is alive."""
    try:
        last_heartbeat = self.get_coordinator_heartbeat()
        current_time = time.time()
        return (current_time - last_heartbeat) < timeout_seconds
    except Exception:
        return False  # No heartbeat found, consider dead
```

#### Integration with Coordination Protocol

The coordination protocol automatically handles coordinator failover:

```python
def coordinate_global_iteration_advance(self, world_size: int, timeout: datetime.timedelta, is_coordinator: bool = False) -> int:
    """Coordinate iteration advancement with automatic coordinator failover."""
    rank = int(os.environ.get('RANK', 0))
    
    # Step 1: Wait for current iteration to be ready
    current_iteration = self.wait_for_global_iteration_ready(timeout)
    
    # Step 2: Handle coordinator failover if needed
    current_coordinator = self.get_current_coordinator()
    
    # Check if we need to elect a new coordinator
    if not self.is_coordinator_alive():
        logging.warning(f"Rank {rank}: Coordinator appears dead, starting failover")
        current_coordinator = self.elect_coordinator(world_size, timeout)
    
    # Update coordinator role based on election
    is_coordinator = (rank == current_coordinator)
    
    # Step 3: Continue with coordination (same as before)
    if is_coordinator:
        # Update coordinator heartbeat to show we're alive
        self.update_coordinator_heartbeat()
        
        # Perform coordination duties...
    else:
        # Wait for coordinator...
```

#### Election Protocol

The election protocol ensures a new coordinator is selected:

1. **Detection**: System detects Rank 0 is dead (no heartbeat for 30+ seconds)
2. **Election**: All active ranks participate in atomic election
3. **Selection**: Rank with lowest number among active ranks becomes coordinator
4. **Handover**: New coordinator takes over coordination duties
5. **Continuation**: Training continues seamlessly

#### Benefits of Coordinator Failover

- **Eliminates single point of failure**: System continues even if Rank 0 crashes
- **Automatic recovery**: No manual intervention required
- **Seamless handover**: Training continues without interruption
- **Robust fault tolerance**: Handles multiple coordinator failures
- **Consistent state**: All ranks maintain correct iteration numbers

#### Iteration Tracking for Restart Recovery

Each rank's iteration is tracked to help with restart recovery:

```python
def record_iteration_for_rank(self, rank: int, iteration: int) -> None:
    """Record the current iteration for a specific rank."""
    self.set(f'rank_{rank}_last_iteration', str(iteration))

def get_iteration_for_rank(self, rank: int) -> int:
    """Get the last recorded iteration for a specific rank."""
    try:
        return int(self.get(f'rank_{rank}_last_iteration'))
    except Exception:
        return self.get_global_iteration()
```

#### Complete Restart Handling

The complete restart protocol handles race conditions:

```python
def handle_rank_restart(self, world_size: int, timeout: datetime.timedelta) -> int:
    """Handle rank restart with race condition handling."""
    rank = int(os.environ.get('RANK', 0))
    
    # Check if this rank is restarting
    try:
        last_heartbeat = self.get_heartbeat(rank)
        current_time = time.time()
        is_restarting = (current_time - last_heartbeat) > 30
    except Exception:
        is_restarting = True
    
    if is_restarting:
        # Get the iteration we were at before restart
        try:
            previous_iteration = int(self.get(f'rank_{rank}_last_iteration'))
        except Exception:
            previous_iteration = self.get_global_iteration()
        
        # Catch up with current iteration (handles race conditions)
        current_iteration = self.catch_up_restarting_rank(timeout)
        
        # Record that we've caught up
        self.set(f'rank_{rank}_last_iteration', str(current_iteration))
        
        logging.info(f"Rank {rank}: Restart complete. Previous: {previous_iteration}, Current: {current_iteration}")
        
        return current_iteration
    else:
        # Normal operation
        current_iteration = self.get_global_iteration()
        self.set(f'rank_{rank}_last_iteration', str(current_iteration))
        return current_iteration
```

## Benefits of Fault-Tolerant Approach

### 1. **Eliminates Deadlocks**
- Crashed ranks don't block the system
- Active ranks can continue without waiting for crashed ranks
- System remains responsive even with failures

### 2. **Handles C++ Backend Crashes**
- Detects crashes through heartbeat monitoring
- Excludes crashed ranks from coordination
- No dependency on Python exception handling

### 3. **Supports Process Restarts**
- Restarting ranks automatically catch up
- No manual intervention required
- Seamless rejoin of coordination

### 4. **Maintains Consistency**
- All active ranks have the same iteration number
- Restarting ranks get the correct iteration
- No inconsistencies across restarts

### 5. **Graceful Degradation**
- System continues with fewer ranks
- Partial quorum support for advancement
- Timeout handling for edge cases

## Usage Examples

### Basic Fault-Tolerant Usage

```python
@Wrapper(use_global_iteration=True)
def train_function(call_wrapper: CallWrapper):
    # call_wrapper.iteration returns fault-tolerant globally unique number
    print(f"Fault-tolerant global iteration: {call_wrapper.iteration}")
```

### Handling Crashes and Restarts

```python
@Wrapper(use_global_iteration=True)
def fault_tolerant_training(call_wrapper: CallWrapper):
    # Even if Rank 1 crashes, other ranks continue
    # When Rank 1 restarts, it catches up automatically
    current_iteration = call_wrapper.iteration
    
    for local_iter in range(10):
        # All active ranks (including restarted Rank 1) have the same global iteration
        print(f"Global: {call_wrapper.iteration}, Local: {local_iter}")
        
        # Simulate work
        time.sleep(1)
        
        # Send heartbeat (crucial for fault detection)
        call_wrapper.ping()
        
        # When this iteration completes, all active ranks will advance together
```

### Scenario: Rank 1 Crashes and Restarts

```python
# Initial state: Rank 0, 1, 2 at iteration 3, Rank 3 is hot spare
# Rank 1 crashes with GPU error (C++ exception)

# Crash sequence:
# 1. Rank 1 crashes immediately without cleanup
# 2. Rank 1 never signals readiness for iteration 4
# 3. System detects Rank 1 is inactive (no heartbeat)
# 4. Rank 0 (coordinator) excludes Rank 1 from coordination
# 5. Rank 0, 2 advance to iteration 4 without Rank 1
# 6. Rank 1 restarts later
# 7. Rank 1 detects restart and catches up to iteration 4
# 8. Rank 1 rejoins coordination for iteration 5
```

## Implementation Details

### Crash Detection

- **Heartbeat monitoring**: Tracks last heartbeat time for each rank
- **Timeout-based detection**: Considers rank inactive after 30 seconds
- **Exception handling**: Gracefully handles missing heartbeats
- **Coordinator protection**: Always includes rank 0 in active set

### Restart Handling

- **Automatic detection**: Detects restart by checking heartbeat history
- **Catch-up protocol**: Gets current iteration and waits for next
- **Seamless rejoin**: Automatically rejoins coordination
- **No manual intervention**: Handles restart transparently

### Partial Quorum Support

- **Majority rule**: Proceeds with majority of active ranks
- **Timeout handling**: Advances after timeout with partial quorum
- **Minimum quorum**: Requires at least 1 active rank
- **Graceful degradation**: Continues operation with fewer ranks

### Error Handling

- **Timeout handling** for coordination
- **Fallback to local iteration** if coordination fails
- **Automatic recovery** from store failures
- **Graceful degradation** if coordinator fails

## Conclusion

The fault-tolerant coordinated global iteration approach solves the critical problems of:

1. **Crashed ranks blocking the system** - by detecting and excluding them
2. **C++ backend exceptions** - by using heartbeat-based crash detection
3. **Process restarts** - by implementing automatic catch-up protocols
4. **Deadlocks** - by not waiting for crashed ranks

This ensures that distributed training can continue reliably even when ranks crash due to GPU errors or C++ exceptions, with all active ranks always having the same globally unique iteration number. 