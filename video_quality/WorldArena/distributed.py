"""Small torch.distributed helpers shared by the retained metrics."""

from datetime import timedelta
import os

import torch
import torch.distributed


def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if torch.distributed.is_initialized():
        # torchrun may launch multiple workers per visible GPU. Map local ranks
        # round-robin onto the selected CUDA devices.
        return torch.device("cuda", get_local_rank() % torch.cuda.device_count())
    return torch.device("cuda", 0)


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def dist_init():
    """Initialize a process group and bind this process to LOCAL_RANK."""
    if torch.distributed.is_initialized():
        return False

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    use_cuda = torch.cuda.is_available()
    device_id = None
    if use_cuda:
        device_index = get_local_rank() % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        device_id = torch.device("cuda", device_index)
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    shares_cuda_devices = use_cuda and local_world_size > torch.cuda.device_count()
    # NCCL expects exclusive GPU ownership per process. When several workers
    # intentionally share each GPU, use Gloo only for the small object
    # collectives; model inference still runs on CUDA.
    backend = (
        "nccl"
        if use_cuda and os.name != "nt" and not shares_cuda_devices
        else "gloo"
    )
    if backend != "nccl":
        device_id = None
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        device_id=device_id,
        # One shard may contain a much longer video than another. Keep early
        # ranks alive while they wait at the final result collective.
        timeout=timedelta(hours=24),
    )
    return True


def dist_init_from_env():
    """Initialize only for a multi-process torchrun invocation."""
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    return dist_init()


def dist_cleanup():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def all_gather(data):
    """Gather arbitrary picklable data from every rank."""
    if get_world_size() == 1:
        return [data]
    gathered = [None] * get_world_size()
    torch.distributed.all_gather_object(gathered, data)
    return gathered


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def merge_list_of_list(results):
    return [item for sublist in results for item in sublist]


def gather_list_of_dict(results):
    return merge_list_of_list(all_gather(results))


def gather_dict(results):
    merged = {}
    for partial in all_gather(results):
        overlap = set(merged) & set(partial)
        if overlap:
            raise ValueError(
                f"Distributed result contains duplicate keys: {sorted(overlap)}"
            )
        merged.update(partial)
    return merged


def distribute_list_to_rank(data_list):
    return data_list[get_rank()::get_world_size()]
