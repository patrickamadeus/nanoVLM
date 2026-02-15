import torch


class _CompiledBlockWrapper(torch.nn.Module):
    def __init__(self, block: torch.nn.Module, *, fullgraph: bool):
        super().__init__()
        self._orig_mod = block
        self._compiled_forward = torch.compile(block.forward, fullgraph=fullgraph)

    def forward(self, *args, **kwargs):
        return self._compiled_forward(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            if name.startswith("_"):
                raise exc
            orig_mod = super().__getattr__("_orig_mod")
            return getattr(orig_mod, name)


def _compile_block_module(block, *, fullgraph: bool):
    if isinstance(block, _CompiledBlockWrapper):
        return block, False
    if not isinstance(block, torch.nn.Module):
        raise TypeError(
            f"Expected torch.nn.Module block, got {type(block).__name__}"
        )
    return _CompiledBlockWrapper(block, fullgraph=fullgraph), True


def _compile_modulelist_blocks(module_list, *, fullgraph: bool):
    compiled_count = 0
    for idx, block in enumerate(module_list):
        compiled_block, was_compiled = _compile_block_module(
            block, fullgraph=fullgraph
        )
        if was_compiled:
            module_list[idx] = compiled_block
            compiled_count += 1
    return compiled_count


def _apply_regional_compile(model):
    compile_fullgraph = True
    summary = {
        "strategy": "regional_submodule_compile_fullgraph",
        "fullgraph": compile_fullgraph,
        "compiled": {},
    }

    if "DualTower" in model.__class__.__name__:
        summary["compiled"]["left_tower_decoder_blocks"] = _compile_modulelist_blocks(
            model.left_tower.decoder.blocks,
            fullgraph=compile_fullgraph,
        )
        summary["compiled"]["right_tower_decoder_blocks"] = _compile_modulelist_blocks(
            model.right_tower.blocks,
            fullgraph=compile_fullgraph,
        )
        # Keep vision/projector eager: packed-image count varies by batch and can
        # trigger repeated recompiles on image-batch dimension.
        summary["compiled"]["left_tower_vision_blocks"] = 0
        summary["compiled"]["left_tower_mp"] = 0
        return model, summary
    elif "VisionLanguageModel" in model.__class__.__name__:
        summary["compiled"]["decoder_blocks"] = _compile_modulelist_blocks(
            model.decoder.blocks,
            fullgraph=compile_fullgraph,
        )
        summary["compiled"]["vision_blocks"] = 0
        summary["compiled"]["mp"] = 0
        return model, summary

    raise ValueError(
        f"Unsupported model type for regional compile: {type(model).__name__}"
    )


def _maybe_mark_batch_dynamic(*, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
    maybe_mark_dynamic = getattr(torch._dynamo, "maybe_mark_dynamic", None)
    tensors = (input_ids, labels, attention_mask)
    for tensor in tensors:
        if tensor.dim() < 2:
            continue
        if maybe_mark_dynamic is not None:
            maybe_mark_dynamic(tensor, 0)
            maybe_mark_dynamic(tensor, 1)
        else:
            torch._dynamo.mark_dynamic(tensor, 0)
            torch._dynamo.mark_dynamic(tensor, 1)
