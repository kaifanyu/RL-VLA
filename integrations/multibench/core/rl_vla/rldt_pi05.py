"""Differentiable, action-expert-only OpenPI pi0.5 backend for RLDT.

This module deliberately does not use ``Policy.infer`` or ``module_jit``:
those inference APIs capture a frozen copy of the parameters.  Optimizer
parameters are explicit arguments to :meth:`Pi05Flow.velocity` instead.
OpenPI's convention is t=1 for Gaussian noise and t=0 for clean actions;
its velocity is noise-minus-data and an Euler step subtracts v / T.

Heavy optional imports happen on construction.  Importing this module does
not initialize JAX, download a checkpoint, or allocate a GPU.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def is_expert_parameter(path: tuple, *, projection_only: bool = False) -> bool:
    """Select action/time projections and expert 1, never VLM expert 0.

    Attention projections share the ``attn`` parent with the frozen VLM.
    Checking only the parent module would silently freeze those projections.
    """
    names = tuple(str(part) for part in path)
    if not names:
        return False
    projections = {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}
    if names[0] in projections:
        return True
    return (
        not projection_only
        and names[:2] == ("PaliGemma", "llm")
        and any(name.endswith("_1") for name in names[2:])
    )


class Pi05Flow:
    """Native JAX pi05_libero checkpoint with a frozen vision/language model.

    ``params`` contains float32 optimizer weights.  ``frozen`` contains the
    remaining bfloat16 model weights.  Every array in a prepared context has
    its batch dimension FIRST, including the KV cache (unlike OpenPI's native
    layer-first cache).  Thus particles can repeat context with an ordinary
    ``jax.tree.map(lambda x: jnp.repeat(x, K, axis=0), context)``.

    ``features`` / ``critic_features`` concatenate masked mean final VLM
    hidden states and the checkpoint-normalized, padded proprioceptive state.
    They are frozen representations; RLDT's critic is trained separately.
    """

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Pi05Flow:
        """Load the local JAX checkpoint, without modifying upstream OpenPI.

        Keys: ``checkpoint``, ``model_config`` (``pi05_libero``), and
        ``projection_only`` (False).  Projection-only is a low-memory software
        diagnostic, not the full action-expert finetuning experiment.
        """
        try:
            import jax
            import jax.numpy as jnp
            from flax import nnx
            from openpi import transforms
            from openpi.models import gemma
            from openpi.models import model as model_module
            from openpi.training import checkpoints
            from openpi.training import config as config_module
        except ImportError as exc:
            raise ImportError(
                "Pi05Flow requires the pinned OpenPI JAX environment. Use "
                "third_party/openpi/.venv/bin/python with this repository's src on PYTHONPATH."
            ) from exc

        model_name = "pi05_base"
        benchmark = config["benchmark"]
        from multibench.policy_spec import ROOT, build_transforms, model_config as base_config
        checkpoint = Path(config.get("checkpoint", ROOT / "checkpoints/openpi/pi05_base")).resolve()
        if not (checkpoint / "params").is_dir():
            raise FileNotFoundError(checkpoint)
        projection_only = config.get("projection_only", False)
        if not isinstance(projection_only, bool):
            raise TypeError("projection_only must be a boolean")

        model_config = base_config()
        if not model_config.pi05:
            raise ValueError("RLDT Pi05Flow requires a pi0.5 model configuration")
        model = model_config.load(
            model_module.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
        )
        model.eval()
        trainable_filter = nnx.All(
            nnx.Param, lambda path, _: is_expert_parameter(path, projection_only=projection_only)
        )
        graphdef, params, frozen = nnx.split(model, trainable_filter, ...)
        params = params.map(lambda _, value: value.replace(value.value.astype(jnp.float32)))
        if not params.flat_state():
            raise ValueError("The checkpoint has no recognized trainable action-expert parameters")

        input_transform, output_transform, calibration = build_transforms(benchmark, model_config)
        self = cls()
        self.params = params
        self.frozen = frozen
        self.graphdef = graphdef
        self.checkpoint = checkpoint
        self.model_config_name = model_name
        self.projection_only = projection_only
        self.horizon = int(model_config.action_horizon)
        self.action_dim = int(model_config.action_dim)
        self.motor_dim = calibration["motor_dim"]
        self.critic_feature_dim = (
            gemma.get_config(model_config.paligemma_variant).width + self.action_dim
        )
        self.feature_dim = self.critic_feature_dim
        self._input_transform = input_transform
        self._output_transform = output_transform
        self._encode_jit = jax.jit(self._encode_prefix)
        asset_records = [
            (
                str(path.relative_to(checkpoint / "assets")),
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in sorted((checkpoint / "assets").rglob("*"))
            if path.is_file()
        ]
        self.metadata = {
            "backend": "openpi-jax-pi05",
            "calibration": calibration,
            "model_config": model_name,
            "checkpoint": str(checkpoint),
            "projection_only": projection_only,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "motor_dim": self.motor_dim,
            "critic_feature_dim": self.critic_feature_dim,
            "trainable_parameters": int(
                sum(np.prod(v.value.shape) for v in params.flat_state().values())
            ),
            "frozen_parameters": int(
                sum(np.prod(v.value.shape) for v in frozen.flat_state().values())
            ),
            "flow_time_convention": "noise_at_1_data_at_0; x_next=x-v/T",
            "feature_extractor": "masked_mean_final_vlm_hidden+normalized_padded_state",
            "checkpoint_metadata_sha256": hashlib.sha256(
                (checkpoint / "params/_METADATA").read_bytes()
            ).hexdigest(),
            "checkpoint_manifest_sha256": hashlib.sha256(
                (checkpoint / "params/manifest.ocdbt").read_bytes()
            ).hexdigest()
            if (checkpoint / "params/manifest.ocdbt").is_file()
            else None,
            "normalization_assets_sha256": hashlib.sha256(
                json.dumps({"base_assets": asset_records, "calibration": calibration}, sort_keys=True).encode()
            ).hexdigest(),
        }
        return self

    def _transform_observation(self, raw: dict[str, Any]) -> dict[str, Any]:
        payload = raw.get("openpi", raw)
        if not isinstance(payload, dict):
            raise TypeError("Expected an OpenPI observation dictionary under observation['openpi']")
        return self._input_transform(copy.deepcopy(payload))

    def prepare(self, raw_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Normalize observations, encode frozen VLM once, and detach context.

        Image preprocessing exactly follows the checkpoint's input transforms.
        The LIBERO environment must already supply its 180-degree image rotation,
        exactly as for ordinary OpenPI inference.
        """
        import jax
        import jax.numpy as jnp
        from openpi.models.model import Observation

        if not raw_batch:
            raise ValueError("Cannot prepare an empty observation batch")
        transformed = [self._transform_observation(raw) for raw in raw_batch]
        batch = jax.tree.map(lambda *values: jnp.asarray(np.stack(values)), *transformed)
        observation = Observation.from_dict(batch)
        return jax.tree.map(
            jax.lax.stop_gradient, self._encode_jit(self.params, self.frozen, observation)
        )

    def _encode_prefix(self, params, frozen, observation):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models import model as model_module
        from openpi.models.pi0 import make_attn_mask

        # Keep checkpoint arrays as explicit JIT inputs. Closing over several
        # GB of VLM weights embeds them into HLO and makes compilation costly.
        model = nnx.merge(self.graphdef, params, frozen)
        observation = model_module.preprocess_observation(None, observation, train=False)
        tokens, mask, autoregressive_mask = model.embed_prefix(observation)
        positions = jnp.cumsum(mask, axis=1) - 1
        (prefix, _), cache = model.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, autoregressive_mask), positions=positions
        )
        weights = mask.astype(jnp.float32)[..., None]
        pooled = jnp.sum(prefix.astype(jnp.float32) * weights, axis=1) / jnp.maximum(
            jnp.sum(weights, axis=1), 1.0
        )
        features = jnp.concatenate([pooled, observation.state.astype(jnp.float32)], axis=-1)
        # Native KV arrays are [layer,batch,token,head,dim].  Public contexts
        # keep batch first so replay sampling / particle replication is uniform.
        return {
            "kv_cache": jax.tree.map(lambda array: jnp.swapaxes(array, 0, 1), cache),
            "prefix_mask": mask,
            "state": observation.state,
            "features": features,
            "critic_features": features,
        }

    @staticmethod
    def repeat_context(context: dict[str, Any], repeats: int) -> dict[str, Any]:
        """Repeat each observation K times in B-major / particle-minor order."""
        import jax
        import jax.numpy as jnp

        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        return jax.tree.map(lambda value: jnp.repeat(value, repeats, axis=0), context)

    def velocity(self, params, context: dict[str, Any], x, t, *, frozen=None):
        """Differentiable noise-minus-data velocity, shape [B,H,32].

        Only the small action expert executes here.  The frozen prefix cache
        already contains all vision/language computation, and ``params`` is
        explicitly passed so optimizer updates immediately affect inference.
        Outer JIT callers must also pass ``frozen`` as a dynamic argument to
        avoid embedding frozen expert weights as constants in projection mode.
        """
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models.model import Observation
        from openpi.models.pi0 import make_attn_mask

        if x.ndim != 3 or x.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"Expected x [B,{self.horizon},{self.action_dim}], got {x.shape}")
        if t.shape != (x.shape[0],):
            raise ValueError(f"Expected timestep [B], got {t.shape}")
        model = nnx.merge(self.graphdef, params, self.frozen if frozen is None else frozen)
        observation = Observation(images={}, image_masks={}, state=context["state"])
        tokens, suffix_mask, autoregressive_mask, condition = model.embed_suffix(observation, x, t)
        prefix_mask = context["prefix_mask"]
        full_mask = jnp.concatenate(
            [
                jnp.broadcast_to(
                    prefix_mask[:, None, :], (x.shape[0], tokens.shape[1], prefix_mask.shape[1])
                ),
                make_attn_mask(suffix_mask, autoregressive_mask),
            ],
            axis=-1,
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        cache = jax.tree.map(lambda array: jnp.swapaxes(array, 0, 1), context["kv_cache"])
        (_, suffix), _ = model.PaliGemma.llm(
            [None, tokens],
            mask=full_mask,
            positions=positions,
            kv_cache=cache,
            adarms_cond=[None, condition],
        )
        return model.action_out_proj(suffix[:, -self.horizon :]).astype(jnp.float32)

    def sample(self, params, context: dict[str, Any], key, steps: int = 10, *, frozen=None):
        """Return detached (final_actions, pre_step_path[T,B,H,D]).

        At path index i the time is ``1 - i / steps``.  All N steps are
        ordinary frozen rollout computation; RLDT separately reevaluates the
        velocity at selected detached states to construct its actor gradient.
        Pass ``frozen`` explicitly through any outer JIT boundary.
        """
        import jax
        import jax.numpy as jnp

        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        noise = jax.random.normal(key, (context["state"].shape[0], self.horizon, self.action_dim))

        def step(x, index):
            t = jnp.full((x.shape[0],), 1.0 - index / steps, dtype=jnp.float32)
            velocity = self.velocity(params, context, x, t, frozen=frozen)
            return x - velocity / steps, x

        final, path = jax.lax.scan(step, noise, jnp.arange(steps, dtype=jnp.float32))
        return jax.lax.stop_gradient(final), jax.lax.stop_gradient(path)

    def decode(self, actions, raw_batch: list[dict[str, Any]]) -> np.ndarray:
        """Apply original checkpoint output transforms and return [B,H,7].

        This is the environment boundary (NumPy, not differentiable).  RLDT
        replay and critic gradients must remain in normalized model coordinates.
        """
        values = np.asarray(actions, dtype=np.float32)
        expected = (len(raw_batch), self.horizon, self.action_dim)
        if values.shape != expected or not np.isfinite(values).all():
            raise ValueError(
                f"Expected finite normalized actions with shape {expected}, got {values.shape}"
            )
        result = []
        for action, raw in zip(values, raw_batch, strict=True):
            inputs = self._transform_observation(raw)
            decoded = self._output_transform({"state": inputs["state"], "actions": action.copy()})
            output = np.asarray(decoded["actions"], dtype=np.float32)
            if output.shape != (self.horizon, self.motor_dim) or not np.isfinite(output).all():
                raise ValueError("Checkpoint output transforms returned invalid LIBERO actions")
            result.append(output)
        return np.stack(result)

    def save(self, output_dir: str | Path, params) -> Path:
        """Save portable action-expert weights without duplicating the VLM.

        The manifest records the required base checkpoint.  Use export_native
        explicitly to produce a self-contained checkpoint for serve_policy.py.
        """
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        target = output / "expert.npz"
        if target.exists() or (output / "rldt_pi05.json").exists():
            raise FileExistsError(f"Refusing to overwrite an existing pi0.5 checkpoint: {output}")
        arrays = {
            "/".join(map(str, path)): np.asarray(value.value, dtype=np.float32)
            for path, value in params.flat_state().items()
        }
        with target.open("xb") as stream:
            np.savez(stream, **arrays)
        (output / "rldt_pi05.json").write_text(
            json.dumps({**self.metadata, "format": "rldt-pi05-expert-v1"}, indent=2) + "\n"
        )
        return target

    def load_params(self, path: str | Path):
        """Restore expert weights from save() or export_native() with provenance.

        Returns a new NNX State; callers assign it to their training state.  The
        frozen VLM is always the base checkpoint loaded by from_config(), so an
        RLDT manifest proving the same base and normalization assets is required.
        To load a generic native checkpoint, set policy.checkpoint before calling
        from_config(); that route loads its complete VLM as well as its expert.
        """
        import jax.numpy as jnp
        from flax import traverse_util
        from openpi.models.model import restore_params

        source = Path(path).expanduser().resolve()
        if source.is_dir() and (source / "params").is_dir():
            self._validate_checkpoint_manifest(source, "openpi-native-rldt-v1")
            flat = traverse_util.flatten_dict(
                restore_params(source / "params", restore_type=np.ndarray)
            )
            arrays = {"/".join(map(str, key)): value for key, value in flat.items()}
        else:
            if source.is_dir():
                source = source / "expert.npz"
            self._validate_checkpoint_manifest(source.parent, "rldt-pi05-expert-v1")
            with np.load(source, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in archive.files}
            expected_keys = {"/".join(map(str, key)) for key in self.params.flat_state()}
            if set(arrays) != expected_keys:
                raise ValueError(
                    "Expert checkpoint parameter keys do not match the configured trainable subset"
                )

        def restore(path, value):
            key = "/".join(map(str, path))
            if key not in arrays:
                raise ValueError(f"Expert checkpoint is missing parameter {key}")
            array = arrays[key]
            if array.shape != value.value.shape or not np.isfinite(array).all():
                raise ValueError(
                    f"Invalid shape or nonfinite parameter in expert checkpoint: {key}"
                )
            return value.replace(jnp.asarray(array, dtype=jnp.float32))

        return self.params.map(restore)

    def _validate_checkpoint_manifest(self, directory: Path, expected_format: str) -> None:
        """Check base identity before reading any potentially large weight file."""
        manifest = directory / "rldt_pi05.json"
        if not manifest.is_file():
            raise ValueError(
                f"Expert loading requires RLDT provenance in {manifest}. "
                "Load a generic native checkpoint via policy.checkpoint instead."
            )
        saved = json.loads(manifest.read_text())
        if not isinstance(saved, dict) or saved.get("format") != expected_format:
            raise ValueError(
                f"Expected checkpoint format {expected_format!r} in {manifest}. "
                "Load a generic native checkpoint via policy.checkpoint instead."
            )
        for key in (
            "model_config",
            "projection_only",
            "checkpoint_metadata_sha256",
            "checkpoint_manifest_sha256",
            "normalization_assets_sha256",
        ):
            if key not in saved or saved[key] != self.metadata[key]:
                raise ValueError(
                    f"Expert checkpoint {key} does not match the loaded base checkpoint"
                )

    def export_native(self, output_dir: str | Path, params) -> Path:
        """Write params/ and assets/ accepted by upstream OpenPI serving.

        This export includes the entire frozen VLM and therefore needs several
        GB of disk space.  Existing output directories are never overwritten.
        """
        import orbax.checkpoint as ocp
        from flax import nnx

        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=False)
        model = nnx.merge(self.graphdef, params, self.frozen)
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(output / "params", {"params": nnx.state(model).to_pure_dict()})
        shutil.copytree(self.checkpoint / "assets", output / "assets")
        (output / "rldt_pi05.json").write_text(
            json.dumps({**self.metadata, "format": "openpi-native-rldt-v1"}, indent=2) + "\n"
        )
        return output
