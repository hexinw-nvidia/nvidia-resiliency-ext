#!/usr/bin/env python3
"""
External TCPStore Service

This service runs TCPStore independently of training processes to solve
the barrier problem across process restarts. The service persists across
rank restarts and provides a stable store for distributed training.
"""

import argparse
import logging
import os
import signal
import sys
import time
from typing import Optional

import torch.distributed as dist


class TCPStoreService:
    """
    External TCPStore service that runs independently of training processes.

    This service provides a persistent TCPStore that survives rank restarts,
    solving the barrier problem where ranks get stuck waiting on old stores.
    """

    def __init__(
        self,
        host: str,
        port: int,
        world_size: int,
        timeout: int = 300,
        use_libuv: bool = True,
        log_level: str = "INFO",
    ):
        self.host = host
        self.port = port
        self.world_size = world_size
        self.timeout = timeout
        self.use_libuv = use_libuv
        self.store = None
        self.running = False

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            level=getattr(logging, log_level.upper()),
        )
        self.logger = logging.getLogger(__name__)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)

    def start(self):
        """Start the TCPStore service."""
        self.logger.info(f"Starting TCPStore service on {self.host}:{self.port}")
        self.logger.info(f"World size: {self.world_size}, Timeout: {self.timeout}s")

        try:
            # Create TCPStore server
            self.store = dist.TCPStore(
                host_name=self.host,
                port=self.port,
                world_size=self.world_size,
                is_master=True,
                timeout=time.time() + self.timeout,
                multi_tenant=True,
                use_libuv=self.use_libuv,
                wait_for_workers=False,  # Don't wait for workers to start
            )

            self.running = True
            self.logger.info("TCPStore service started successfully")

            # Keep the service running
            self._run()

        except Exception as e:
            self.logger.error(f"Failed to start TCPStore service: {e}")
            raise

    def _run(self):
        """Main service loop."""
        self.logger.info("TCPStore service is running. Press Ctrl+C to stop.")

        try:
            while self.running:
                time.sleep(1)

                # Optional: Add health checks or monitoring here
                # For example, check if store is still responsive

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            self.stop()

    def stop(self):
        """Stop the TCPStore service."""
        if self.running:
            self.logger.info("Stopping TCPStore service...")
            self.running = False

            if self.store is not None:
                try:
                    # Note: TCPStore doesn't have a clean shutdown method
                    # The service will be cleaned up when the process exits
                    del self.store
                except:
                    pass
                self.store = None

            self.logger.info("TCPStore service stopped")

    def get_connection_info(self) -> dict:
        """Get connection information for clients."""
        return {
            'host': self.host,
            'port': self.port,
            'world_size': self.world_size,
            'timeout': self.timeout,
            'use_libuv': self.use_libuv,
        }


def main():
    """Main entry point for the external TCPStore service."""
    parser = argparse.ArgumentParser(description="External TCPStore Service")
    parser.add_argument(
        '--host', default='0.0.0.0', help='Host to bind the TCPStore server to (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=29500,
        help='Port to bind the TCPStore server to (default: 29500)',
    )
    parser.add_argument(
        '--world-size',
        type=int,
        required=True,
        help='Number of workers that will connect to this store',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds for store operations (default: 300)',
    )
    parser.add_argument(
        '--use-libuv',
        action='store_true',
        default=True,
        help='Use libuv backend for better performance (default: True)',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)',
    )

    args = parser.parse_args()

    # Create and start the service
    service = TCPStoreService(
        host=args.host,
        port=args.port,
        world_size=args.world_size,
        timeout=args.timeout,
        use_libuv=args.use_libuv,
        log_level=args.log_level,
    )

    try:
        service.start()
    except Exception as e:
        logging.error(f"Service failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
