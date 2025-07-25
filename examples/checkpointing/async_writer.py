import argparse
import logging
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed import checkpoint
from torch.distributed.checkpoint import DefaultLoadPlanner, DefaultSavePlanner, FileSystemReader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, DistributedSampler

from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue, AsyncRequest
from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import FileSystemWriterAsync
from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
    init_checkpoint_metadata_cache,
    save_state_dict_async_finalize,
    save_state_dict_async_plan,
)

# Set up basic logging configuration
# Try setting `DEBUG` to see detailed steps of NVRx checkpointing
logging.basicConfig(level=logging.INFO)

FEAT_SIZE = 4096
DNN_OUT_SIZE = 128
BATCH_SIZE = 100
NUM_EPOCHS = 10
DATASET_LEN = 100000
CKPT_INTERVAL = 100


def print_on_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


def parse_args():
    parser = argparse.ArgumentParser(description="Async Checkpointing Example")
    parser.add_argument(
        "--ckpt_dir",
        default="/tmp/test_checkpointing/",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--thread_count",
        default=2,
        type=int,
        help="Threads to use during saving. Affects the number of files in the checkpoint (saving ranks * num_threads).",
    )
    parser.add_argument(
        '--no_persistent_queue',
        action='store_false',
        default=True,
        dest='persistent_queue',
        help=(
            "Disables a persistent version of AsyncCallsQueue. "
            "Effective only when --async_save is set."
        ),
    )

    return parser.parse_args()


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.rand((FEAT_SIZE,), dtype=torch.float32, device='cuda')
        y = torch.rand((DNN_OUT_SIZE,), dtype=torch.float32, device='cuda')
        return x, y


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(FEAT_SIZE, FEAT_SIZE)
        self.fc2 = nn.Linear(FEAT_SIZE, DNN_OUT_SIZE)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


def init_distributed_backend(backend="nccl"):
    """Initializes the distributed process group using the specified backend."""
    try:
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        logging.info(f"Process {rank} initialized with {backend} backend.")
        return rank, torch.distributed.get_world_size()
    except Exception as e:
        logging.error(f"Failed to initialize distributed backend: {e}")
        raise


def get_save_and_finalize_callbacks(writer, save_state_dict_ret) -> AsyncRequest:
    """Creates an async save request with a finalize function."""
    save_fn, preload_fn, save_args = writer.get_save_function_and_args()

    def finalize_fn():
        """Finalizes async checkpointing and synchronizes processes."""
        save_state_dict_async_finalize(*save_state_dict_ret)
        dist.barrier()

    return AsyncRequest(save_fn, save_args, [finalize_fn], preload_fn=preload_fn)


def save_checkpoint(checkpoint_dir, async_queue, model, thread_count):
    """Asynchronously saves a model checkpoint."""
    state_dict = model.state_dict()
    planner = DefaultSavePlanner()
    writer = FileSystemWriterAsync(checkpoint_dir, thread_count=thread_count)
    coordinator_rank = 0

    save_state_dict_ret = save_state_dict_async_plan(
        state_dict, writer, None, coordinator_rank, planner=planner, enable_cache=True
    )
    save_request = get_save_and_finalize_callbacks(writer, save_state_dict_ret)
    async_queue.schedule_async_request(save_request)


def load_checkpoint(checkpoint_dir, model):
    """Loads a model checkpoint synchronously."""
    state_dict = model.state_dict()
    checkpoint.load(
        state_dict=state_dict,
        storage_reader=FileSystemReader(checkpoint_dir),
        planner=DefaultLoadPlanner(),
    )
    return state_dict


def main():
    args = parse_args()
    logging.info(f"Arguments: {args}")

    # Initialize distributed training
    rank, world_size = init_distributed_backend(backend="nccl")

    # Define checkpoint directory based on iteration number
    dataset = SimpleDataset(size=DATASET_LEN)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

    # Model, optimizer, and FSDP wrapper
    model = SimpleModel().to("cuda")
    fsdp_model = FSDP(model)
    optimizer = optim.SGD(fsdp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Create an async queue for handling asynchronous operations
    async_queue = AsyncCallsQueue(persistent=args.persistent_queue)

    iteration = 0
    num_iters_in_epoch = len(dataloader)
    print_on_rank0(f"num_iters_in_epoch: {num_iters_in_epoch}")

    num_iters_for_10pct = num_iters_in_epoch // 10  # iters for 1/10 of epoch
    checkpoint_dir = None
    sampler.set_epoch(0)

    init_checkpoint_metadata_cache()

    for batch_idx, (data, target) in enumerate(dataloader):
        async_queue.maybe_finalize_async_calls(blocking=False, no_dist=False)
        if (batch_idx % num_iters_for_10pct) == 0 and rank == 0:
            print(f"Epoch 0 progress: {100 * batch_idx / num_iters_in_epoch:.2f}%")
        optimizer.zero_grad()
        output = fsdp_model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % num_iters_for_10pct == 0:
            iteration = batch_idx
            checkpoint_dir = f"{args.ckpt_dir}/iter_{iteration:07d}"
            # Save the model asynchronously
            save_checkpoint(checkpoint_dir, async_queue, fsdp_model, args.thread_count)
            print_on_rank0(f"Checkpoint Save triggered: {checkpoint_dir}, iteration: {iteration}")
            iteration += batch_idx
    print_on_rank0(f"Epoch 0 complete. Loss: {loss.item()}")

    logging.info("Finalizing checkpoint save...")
    async_queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
    async_queue.close()  # Explicitly close queue (optional)

    # Synchronize processes before loading
    dist.barrier()
    print_on_rank0(f"loading from {checkpoint_dir}")
    # Load the checkpoint
    loaded_sd = load_checkpoint(checkpoint_dir, fsdp_model)

    # Synchronize again to ensure all ranks have completed loading
    dist.barrier()

    # Clean up checkpoint directory (only on rank 0)
    if dist.get_rank() == 0:
        logging.info(f"Cleaning up checkpoint directory: {args.ckpt_dir}")
        shutil.rmtree(args.ckpt_dir)

    # Ensure NCCL process group is properly destroyed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
