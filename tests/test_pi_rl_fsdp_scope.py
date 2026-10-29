"""Real CPU FSDP regression for a frozen VLM above trainable action modules."""

import functools
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
from rlinf_bridge.fsdp_scope import ignored_modules_for_native_openpi

NATIVE = {"model_type": "openpi", "enforce_native_scope": True}


class TinyNativeOpenPi(nn.Module):
    """Match the frozen-root/trainable-children topology and mixed dtypes."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pi05=True, train_expert_only=True)
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.paligemma = nn.Linear(4, 4).to(torch.bfloat16)
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.model = nn.Linear(4, 4).to(torch.bfloat16)
        self.paligemma_with_expert.gemma_expert.lm_head = nn.Linear(4, 6).to(torch.bfloat16)
        self.action_out_proj = nn.Linear(4, 2)
        self.value_head = nn.Linear(4, 2)
        for module in (self.paligemma_with_expert.paligemma,
                       self.paligemma_with_expert.gemma_expert.lm_head):
            module.requires_grad_(False)

    def forward(self, observation):
        prefix = self.paligemma_with_expert.paligemma(observation.to(torch.bfloat16))
        expert = self.paligemma_with_expert.gemma_expert.model(prefix)
        return self.action_out_proj(expert.float()) + self.value_head(prefix.float())


def test_scope_is_opt_in_and_rejects_trainable_exclusions():
    assert ignored_modules_for_native_openpi(None, {}, device="cpu") is None
    assert ignored_modules_for_native_openpi(
        None, NATIVE | {"model_type": "other"}, device="cpu") is None
    assert ignored_modules_for_native_openpi(
        None, NATIVE | {"enforce_native_scope": False}, device="cpu") is None
    model = TinyNativeOpenPi()
    model.config.train_expert_only = False
    with pytest.raises(ValueError, match="frozen VLM"):
        ignored_modules_for_native_openpi(model, NATIVE, device="cpu")
    model.config.train_expert_only = True
    model.paligemma_with_expert.paligemma.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="VLM.*already be frozen"):
        ignored_modules_for_native_openpi(model, NATIVE, device="cpu")
    model.paligemma_with_expert.paligemma.weight.requires_grad_(False)
    model.paligemma_with_expert.gemma_expert.lm_head.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="language head.*already be frozen"):
        ignored_modules_for_native_openpi(model, NATIVE, device="cpu")


@pytest.fixture
def process_group(tmp_path):
    if dist.is_initialized():
        raise RuntimeError("This regression must own its CPU process group")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}",
                            rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("accumulate", [False, True])
def test_repeated_microbatches_preserve_frozen_weights_and_checkpoint_keys(
    tmp_path, process_group, accumulate,
):
    torch.manual_seed(5)
    raw = TinyNativeOpenPi()
    initial = {name: value.detach().clone() for name, value in raw.state_dict().items()}
    trainable_names = {name for name, value in raw.named_parameters() if value.requires_grad}
    ignored = ignored_modules_for_native_openpi(raw, NATIVE, device=torch.device("cpu"))
    assert ignored == [raw.paligemma_with_expert.paligemma,
                       raw.paligemma_with_expert.gemma_expert.lm_head]
    assert all(parameter.dtype == torch.bfloat16
               for module in ignored for parameter in module.parameters())
    wrapped_children = {raw.paligemma_with_expert.gemma_expert.model,
                        raw.action_out_proj, raw.value_head}
    policy = functools.partial(lambda_auto_wrap_policy,
                               lambda_fn=lambda module: module in wrapped_children)
    model = FSDP(raw, device_id=torch.device("cpu"),
                 sharding_strategy=ShardingStrategy.NO_SHARD, use_orig_params=True,
                 auto_wrap_policy=policy, ignored_modules=ignored)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        for micro in range(2):
            # Match RLinf's context placement: forward outside, backward inside.
            loss = model(torch.randn(1, 4)).square().mean() / 2
            with (model.no_sync() if accumulate and micro == 0 else nullcontext()):
                loss.backward()
        optimizer.step()

    state = model.state_dict()
    assert state.keys() == initial.keys()
    for name in initial.keys() - trainable_names:
        torch.testing.assert_close(state[name], initial[name], rtol=0, atol=0)
    expert_name = "paligemma_with_expert.gemma_expert.model.weight"
    assert not torch.equal(state[expert_name], initial[expert_name])
    assert not torch.equal(state["value_head.weight"], initial["value_head.weight"])
    assert all(torch.isfinite(value).all() for value in state.values())

    checkpoint = tmp_path / "full_weights.pt"
    torch.save(state, checkpoint)
    fresh = TinyNativeOpenPi()
    fresh.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    for name, value in fresh.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    with torch.no_grad():
        probe = torch.randn(1, 4)
        torch.testing.assert_close(model(probe), fresh(probe), rtol=0, atol=0)
