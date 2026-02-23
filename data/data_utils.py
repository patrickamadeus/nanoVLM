import torch
import torch.distributed as dist
from collections.abc import Iterator

try:
    from requests.exceptions import ChunkedEncodingError, ConnectionError as RequestsConnectionError, ReadTimeout
except Exception:  # pragma: no cover
    ChunkedEncodingError = tuple()  # type: ignore[assignment]
    RequestsConnectionError = tuple()  # type: ignore[assignment]
    ReadTimeout = tuple()  # type: ignore[assignment]


def _is_recoverable_stream_error(exc: Exception) -> bool:
    recoverable_types = (
        TimeoutError,
        ConnectionResetError,
        BrokenPipeError,
        ChunkedEncodingError,
        RequestsConnectionError,
        ReadTimeout,
    )
    if isinstance(exc, recoverable_types):
        return True

    msg = str(exc)
    keywords = (
        "ChunkedEncodingError",
        "IncompleteRead",
        "Connection broken",
        "ReadTimeout",
        "RemoteDisconnected",
        "Max retries exceeded",
        "temporarily unavailable",
        "Connection aborted",
    )
    if any(k in msg for k in keywords):
        return True

    root = exc.__cause__ or exc.__context__
    if root is not None and root is not exc:
        return _is_recoverable_stream_error(root)
    return False


def _is_batch_valid(batch):
    """
    Check if a batch is valid for training/evaluation.
    A valid batch must have input_ids and at least one image.
    """
    if not batch:
        return False
    # The collator can return a batch with empty lists
    if len(batch['input_ids']) == 0:
        return False
    
    if len(batch['images']) == 0:
        return False
    
    # `images` is a list of lists of tensors. Check that at least one image is not None.
    if len([img for sublist in batch['images'] for img in sublist]) == 0:
        # During training, not having images creates gradients computed without all model parameters.
        # This creates deadlocks in DDP.
        return False

    return True


def synchronized_dataloader_step(train_loader, is_dist):
    """
    Create a synchronized iterator that handles uneven data distribution in DDP.
    All ranks will stop when the first rank runs out of data.
    This happens because when packing a presharded dataset, a rank might have less groups than the others.
    It also handles cases where a collator returns an empty/invalid batch on some ranks,
    by ensuring all ranks skip the invalid batch and attempt to fetch a new one.
    """
    max_consecutive_stream_errors = 100
    consecutive_stream_errors = 0

    if not is_dist:
        # For single GPU, keep training on transient streaming failures.
        if isinstance(train_loader, Iterator):
            train_iter = train_loader
        else:
            train_iter = iter(train_loader)

        while True:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            except Exception as exc:
                if _is_recoverable_stream_error(exc):
                    consecutive_stream_errors += 1
                    print(
                        f"[WARN] Recoverable dataloader stream error "
                        f"({consecutive_stream_errors}/{max_consecutive_stream_errors}): {exc}"
                    )
                    if consecutive_stream_errors >= max_consecutive_stream_errors:
                        raise RuntimeError(
                            f"Exceeded {max_consecutive_stream_errors} consecutive recoverable dataloader "
                            f"stream errors. Last error: {exc}"
                        ) from exc
                    continue
                raise

            consecutive_stream_errors = 0
            if _is_batch_valid(batch):
                yield batch
        return
    
    # For DDP, we need synchronization.
    if isinstance(train_loader, Iterator):
        train_iter = train_loader
    else:
        train_iter = iter(train_loader)
    
    status_valid = 2
    status_recoverable_error = 1
    status_eof = 0

    while True:
        batch = None
        local_status = status_valid
        is_valid = False
        try:
            while not is_valid:
                batch = next(train_iter)
                is_valid = _is_batch_valid(batch)
        except StopIteration:
            local_status = status_eof
        except Exception as exc:
            if _is_recoverable_stream_error(exc):
                local_status = status_recoverable_error
                consecutive_stream_errors += 1
                if dist.get_rank() == 0:
                    print(
                        f"[WARN] Recoverable dataloader stream error "
                        f"({consecutive_stream_errors}/{max_consecutive_stream_errors}): {exc}"
                    )
                if consecutive_stream_errors >= max_consecutive_stream_errors:
                    raise RuntimeError(
                        f"Exceeded {max_consecutive_stream_errors} consecutive recoverable dataloader "
                        f"stream errors. Last error: {exc}"
                    ) from exc
            else:
                raise

        global_status = torch.tensor(local_status, device=torch.cuda.current_device())
        # We synchronize across all ranks.
        # 0 -> at least one rank EOF (stop all)
        # 1 -> at least one rank hit recoverable stream error (skip step and continue)
        # 2 -> all ranks have valid batch (yield)
        dist.all_reduce(global_status, op=dist.ReduceOp.MIN)

        if global_status.item() == status_eof:
            break
        if global_status.item() == status_recoverable_error:
            continue

        consecutive_stream_errors = 0
        yield batch
    return None
