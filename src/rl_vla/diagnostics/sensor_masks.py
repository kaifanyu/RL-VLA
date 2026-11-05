"""Capture instance masks at the native camera observable's sampling instant."""

import numpy as np

from rl_vla.diagnostics.observation_bank import CAMERAS
from rl_vla.diagnostics.old_target import camera_transform


class SensorMaskCapture:
    """Read-only sensor hooks; never replace the original camera measurement.

    Robosuite observables can sample before a primitive's final physics substep.
    Rendering after env.step therefore need not align with the policy image.
    These hooks render segmentation immediately after each original RGB sensor.
    """

    def __init__(self, env):
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

        self.backend = env.backend.env if hasattr(env.backend, "env") else env.backend
        self.convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        self.enabled = False
        self.frames = {}
        self.originals = {}
        original_setup = self.backend._setup_observables

        def setup_observables():
            observables = original_setup()
            self._install(observables)
            return observables

        self.backend._setup_observables = setup_observables
        self._install(self.backend._observables)

    def _install(self, observables):
        for camera, native in CAMERAS.items():
            observable = observables[f"{native}_image"]
            original = observable._sensor
            self.originals[camera] = original

            def sensor(cache, *, original=original, camera=camera, native=native):
                value = original(cache)
                if self.enabled:
                    sim = self.backend.sim
                    before = sim.get_state().flatten().copy()
                    segmentation = sim.render(camera_name=native, width=256, height=256,
                                              depth=False, segmentation=True)
                    if not np.array_equal(before, sim.get_state().flatten()):
                        raise ValueError("Segmentation rendering mutated native simulator state")
                    self.frames[camera] = {
                        "rgb": np.asarray(value).copy(),
                        "geometry": segmentation[::self.convention, :, 1].copy(),
                        "transform": camera_transform(sim, native),
                        "simulator_time": float(sim.data.time),
                    }
                return value

            sensor.__modality__ = original.__modality__
            observable._sensor = sensor

    def masks(self, observation):
        from PIL import Image

        from rl_vla.envs.libero import _resize_image

        masks, alignment = {}, {}
        owners = self.backend.model.instances_to_ids
        for camera in CAMERAS:
            frame = self.frames[camera]
            rgb = _resize_image(np.ascontiguousarray(frame["rgb"][::-1, ::-1]), 224)
            expected = observation["openpi"][f"observation/{camera}"]
            if not np.array_equal(rgb, expected):
                raise ValueError(f"Native sensor capture does not exactly match policy image: {camera}")
            geometry = frame["geometry"][::-1, ::-1]
            owned = np.zeros((224, 224), bool)
            for instance, groups in owners.items():
                mask = np.isin(geometry, groups.get("geom", []))
                mask = np.asarray(Image.fromarray(mask).resize((224, 224), Image.Resampling.NEAREST), bool)
                masks[f"{camera}__{instance}"] = mask
                owned |= mask
            masks[f"{camera}__unowned_background"] = ~owned
            alignment[camera] = {"exact_sensor_rgb_match": True, "simulator_time": frame["simulator_time"],
                                 "convention": self.convention, "rotation_degrees": 180,
                                 "mask_resize": "nearest", "simulator_state_unchanged": True}
        return masks, {"instances": owners, "camera_alignment": alignment,
                       "mask_source": "MuJoCo visible geometry IDs captured immediately after each native RGB camera sensor; native RGB unchanged"}

    @property
    def transforms(self):
        return {camera: frame["transform"] for camera, frame in self.frames.items()}
