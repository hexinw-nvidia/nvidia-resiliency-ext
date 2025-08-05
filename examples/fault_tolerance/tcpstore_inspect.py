#!/usr/bin/env python3
"""
TCPStore Inspection Script

This script connects to an existing TCPStore service and allows you to inspect
barrier key values and other store contents for debugging purposes.
"""

import argparse
import datetime
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

import torch.distributed as dist


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


class TCPStoreInspector:
    """Utility class to inspect TCPStore contents."""
    
    def __init__(self, host: str, port: int, world_size: int, timeout: int = 30):
        self.host = host
        self.port = port
        self.world_size = world_size
        self.timeout = timeout
        self.store = None
        self.iteration = 0  # Default iteration
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Connect to the TCPStore service."""
        try:
            self.logger.info(f"Connecting to TCPStore at {self.host}:{self.port}")
            
            # Create TCPStore client (not server)
            self.store = dist.TCPStore(
                host_name=self.host,
                port=self.port,
                world_size=self.world_size,
                is_master=False,  # Client mode
                timeout=datetime.timedelta(seconds=self.timeout),
                multi_tenant=True,
                use_libuv=True,
                wait_for_workers=False,
            )
            
            self.logger.info("✅ Successfully connected to TCPStore")
            
            # Test if store is responsive by trying to get a non-existent key
            self.logger.info("Testing store responsiveness...")
            try:
                # Try to get a test key that shouldn't exist
                test_key = f"_test_key_{int(time.time())}"
                self.get_key(test_key)
                self.logger.info("✅ Store is responsive")
            except Exception as e:
                self.logger.warning(f"Store responsiveness test failed: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to TCPStore: {e}")
            return False
    
    def get_key(self, key: str) -> Optional[str]:
        """Get a key value from the store with timeout."""
        self.logger.debug(f"Getting key: '{key}'")
        try:
            # Try to get the key with a very short timeout
            def get_with_timeout():
                try:
                    value = self.store.get(key)
                    return value.decode('utf-8') if value else None
                except Exception as e:
                    if "Key not found" in str(e) or "No such key" in str(e):
                        return None
                    raise e
            
            # Use threading with very short timeout
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = get_with_timeout()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=1)  # 1 second timeout for each get operation
            
            if thread.is_alive():
                # If it's still alive, the key probably doesn't exist
                # TCPStore.get() blocks until the key exists
                return None
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
            
        except Exception as e:
            # Don't log error for non-existent keys, just return None
            if "Key not found" in str(e) or "No such key" in str(e):
                return None
            self.logger.error(f"Failed to get key '{key}': {e}")
            return None
    
    def check_key_exists(self, key: str) -> bool:
        """Check if a key exists without blocking."""
        self.logger.debug(f"Checking if key exists: '{key}'")
        try:
            # Try to get the key with a very short timeout
            def get_with_timeout():
                try:
                    value = self.store.get(key)
                    return True  # If we get here, the key exists
                except Exception as e:
                    if "Key not found" in str(e) or "No such key" in str(e):
                        return False
                    raise e
            
            # Use threading with very short timeout
            result = [False]
            exception = [None]
            
            def target():
                try:
                    result[0] = get_with_timeout()
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=1)  # 1 second timeout
            
            if thread.is_alive():
                # If it's still alive, the key probably doesn't exist
                return False
            
            if exception[0]:
                return False
            
            return result[0]
            
        except Exception:
            return False
    
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys in the store (if supported)."""
        # Note: TCPStore doesn't have a built-in list_keys method
        # This is a placeholder for future implementation
        self.logger.warning("TCPStore doesn't support listing all keys")
        return []
    
    def inspect_barrier(self, barrier_name: str):
        """Inspect a specific barrier."""
        self.logger.info(f"=== Inspecting Barrier: {barrier_name} (Iteration: {self.iteration}) ===")
        
        # Try different function names for barriers
        function_names = ['barrier', 'reentrant_barrier']
        
        self.logger.info(f"Looking for keys with prefix: _inprocess_{self.iteration}")
        self.logger.info(f"Trying function names: {function_names}")
        
        for function_name in function_names:
            self.logger.info(f"Trying function '{function_name}':")
            
            # Construct the correct keys based on the actual implementation
            # The PrefixStore uses '/' as separator, not ':'
            store_key = f"_inprocess_{self.iteration}/inprocess_barrier_prefix:{function_name}:{barrier_name}"
            last_worker_arrived_key = f"{store_key}:last_worker_arrived"
            arrived_key = f"{store_key}:arrived"
            
            self.logger.info(f"  Store key: '{store_key}'")
            self.logger.info(f"  Last worker arrived key: '{last_worker_arrived_key}'")
            self.logger.info(f"  Arrived key: '{arrived_key}'")
            
            # Check if keys exist
            store_exists = self.check_key_exists(store_key)
            last_worker_exists = self.check_key_exists(last_worker_arrived_key)
            arrived_exists = self.check_key_exists(arrived_key)
            
            self.logger.info(f"  Store key exists: {store_exists}")
            self.logger.info(f"  Last worker arrived key exists: {last_worker_exists}")
            self.logger.info(f"  Arrived key exists: {arrived_exists}")
            
            # If any key exists, get the values
            if store_exists or arrived_exists or last_worker_exists:
                if store_exists:
                    store_value = self.get_key(store_key)
                    self.logger.info(f"  Store value: {store_value}")
                
                if arrived_exists:
                    arrived_value = self.get_key(arrived_key)
                    self.logger.info(f"  Arrived value: {arrived_value}")
                
                if last_worker_exists:
                    last_worker_value = self.get_key(last_worker_arrived_key)
                    self.logger.info(f"  Last worker value: {last_worker_value}")
                
                return  # Found keys for this function, no need to try others
        
        # If no keys found with prefix, try without prefix
        self.logger.info("No keys found with prefix, trying without prefix...")
        for function_name in function_names:
            self.logger.info(f"Trying without prefix, function '{function_name}':")
            
            # Try keys without the iteration prefix
            store_key = f"inprocess_barrier_prefix:{function_name}:{barrier_name}"
            last_worker_arrived_key = f"{store_key}:last_worker_arrived"
            arrived_key = f"{store_key}:arrived"
            
            self.logger.info(f"  Store key: '{store_key}'")
            self.logger.info(f"  Last worker arrived key: '{last_worker_arrived_key}'")
            self.logger.info(f"  Arrived key: '{arrived_key}'")
            
            store_exists = self.check_key_exists(store_key)
            last_worker_exists = self.check_key_exists(last_worker_arrived_key)
            arrived_exists = self.check_key_exists(arrived_key)
            
            self.logger.info(f"  Store key exists: {store_exists}")
            self.logger.info(f"  Last worker arrived key exists: {last_worker_exists}")
            self.logger.info(f"  Arrived key exists: {arrived_exists}")
            
            if store_exists or arrived_exists or last_worker_exists:
                if store_exists:
                    store_value = self.get_key(store_key)
                    self.logger.info(f"  Store value: {store_value}")
                
                if arrived_exists:
                    arrived_value = self.get_key(arrived_key)
                    self.logger.info(f"  Arrived value: {arrived_value}")
                
                if last_worker_exists:
                    last_worker_value = self.get_key(last_worker_arrived_key)
                    self.logger.info(f"  Last worker value: {last_worker_value}")
                
                return
        
        self.logger.info("No barrier keys found")
    
    def inspect_common_keys(self):
        """Inspect common keys used by the wrapper."""
        self.logger.info(f"=== Inspecting Common Keys (Iteration: {self.iteration}) ===")
        
        # Common keys that might exist
        common_keys = [
            "any_rank_interrupted",
            "any_rank_completed",
            "interruption_records", 
            "terminated_ranks",
            "active_ranks",
            "initial_ranks",
            "heartbeat_ranks",
        ]
        
        for key_name in common_keys:
            self.logger.info(f"Checking key: '{key_name}'")
            
            # Try with iteration prefix first
            # The PrefixStore uses '/' as separator, not ':'
            prefixed_key = f"_inprocess_{self.iteration}/{key_name}"
            self.logger.info(f"  With prefix: '{prefixed_key}'")
            
            exists = self.check_key_exists(prefixed_key)
            self.logger.info(f"  Exists: {exists}")
            
            if exists:
                value = self.get_key(prefixed_key)
                self.logger.info(f"  Value: {value}")
            else:
                # Try without prefix
                self.logger.info(f"  Without prefix: '{key_name}'")
                exists_no_prefix = self.check_key_exists(key_name)
                self.logger.info(f"  Exists: {exists_no_prefix}")
                
                if exists_no_prefix:
                    value = self.get_key(key_name)
                    self.logger.info(f"  Value: {value}")
            
            print()  # Add spacing between keys
    
    def inspect_all_barriers(self):
        """Inspect all known barrier types."""
        barriers = [
            "initial_barrier",
            "completion_barrier", 
            "iteration_barrier",
            "termination_barrier",
        ]
        
        for barrier in barriers:
            self.inspect_barrier(barrier)
            print()  # Add spacing between barriers
    
    def discover_keys(self):
        """Try to discover what keys actually exist in the store."""
        self.logger.info("=== Discovering Keys in Store ===")
        
        # Test common patterns
        test_patterns = [
            # Common prefixes
            "_inprocess_",
            "inprocess_barrier_prefix",
            "any_rank_",
            "interruption_",
            "terminated_",
            "initial_rank_",
            "active_rank_",
            "heartbeat_",
            "state_",
            "key_",
            
            # Try different iterations
            "_inprocess_0",
            "_inprocess_1", 
            "_inprocess_2",
            
            # Try different barrier patterns
            "barrier:",
            "reentrant_barrier:",
            "initial_barrier",
            "completion_barrier",
            "iteration_barrier",
            "termination_barrier",
        ]
        
        found_keys = []
        
        for pattern in test_patterns:
            # Try to get a key that starts with this pattern
            test_key = f"{pattern}_test"
            try:
                # Use a very short timeout to check if any key with this pattern exists
                def check_pattern():
                    try:
                        # Try to get any key that might match this pattern
                        # We'll use a simple approach: try to get a key and see if it times out
                        self.store.get(test_key)
                        return False  # If we get here, the key doesn't exist
                    except Exception as e:
                        if "Key not found" in str(e) or "No such key" in str(e):
                            return False
                        return True  # Some other error, might indicate the pattern exists
                
                result = [False]
                thread = threading.Thread(target=lambda: result.__setitem__(0, check_pattern()))
                thread.daemon = True
                thread.start()
                thread.join(timeout=0.5)  # Very short timeout
                
                if result[0]:
                    found_keys.append(pattern)
                    self.logger.info(f"  Found pattern: {pattern}")
                    
            except Exception:
                pass
        
        if found_keys:
            self.logger.info(f"Discovered {len(found_keys)} patterns: {found_keys}")
        else:
            self.logger.info("No common patterns found")
        
        # Try to get some basic keys that should always exist if the store is being used
        basic_keys = [
            "any_rank_interrupted",
            "any_rank_completed", 
            "interruption_records",
            "terminated_ranks",
        ]
        
        self.logger.info("Checking basic keys:")
        for key in basic_keys:
            self.logger.info(f"  Testing key: '{key}'")
            exists = self.check_key_exists(key)
            self.logger.info(f"  {key}: {exists}")
            
            # Also try with common prefixes
            for prefix in ["_inprocess_0", "_inprocess_1"]:
                # The PrefixStore uses '/' as separator, not ':'
                prefixed_key = f"{prefix}/{key}"
                self.logger.info(f"  Testing prefixed key: '{prefixed_key}'")
                exists = self.check_key_exists(prefixed_key)
                if exists:
                    self.logger.info(f"  {prefixed_key}: {exists}")
        
        return found_keys
    
    def close(self):
        """Close the connection."""
        if self.store:
            self.store = None
            self.logger.info("Connection closed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TCPStore Inspection Tool")
    parser.add_argument(
        '--host', 
        default='localhost',
        help='TCPStore host (default: localhost)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        required=True,
        help='TCPStore port'
    )
    parser.add_argument(
        '--world-size', 
        type=int, 
        default=16,
        help='World size (default: 16)'
    )
    parser.add_argument(
        '--timeout', 
        type=int, 
        default=30,
        help='Connection timeout in seconds (default: 30)'
    )
    parser.add_argument(
        '--iteration', 
        type=int, 
        default=0,
        help='Iteration number (default: 0)'
    )
    parser.add_argument(
        '--barrier', 
        default='initial_barrier',
        help='Specific barrier to inspect (default: initial_barrier)'
    )
    parser.add_argument(
        '--all-barriers', 
        action='store_true',
        help='Inspect all barriers'
    )
    parser.add_argument(
        '--common-keys', 
        action='store_true',
        help='Inspect common keys'
    )
    parser.add_argument(
        '--all', 
        action='store_true',
        help='Inspect everything'
    )
    parser.add_argument(
        '--discover', 
        action='store_true',
        help='Discover what keys exist in the store'
    )
    
    args = parser.parse_args()
    
    # Create inspector
    inspector = TCPStoreInspector(
        host=args.host,
        port=args.port,
        world_size=args.world_size,
        timeout=args.timeout,
    )
    
    # Set iteration for key construction
    inspector.iteration = args.iteration
    
    try:
        # Connect to store
        if not inspector.connect():
            sys.exit(1)
        
        # Perform inspections based on arguments
        if args.discover:
            inspector.discover_keys()
        elif args.all:
            inspector.inspect_all_barriers()
            inspector.inspect_common_keys()
        elif args.all_barriers:
            inspector.inspect_all_barriers()
        elif args.common_keys:
            inspector.inspect_common_keys()
        else:
            inspector.inspect_barrier(args.barrier)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        inspector.close()


if __name__ == '__main__':
    main() 