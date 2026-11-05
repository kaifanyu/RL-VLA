"""Read-only native object/gripper measurements for reviewable failure analysis."""

import gymnasium as gym
import numpy as np


class ObjectTelemetryEnv(gym.Wrapper):
    """Decorate info only; never modify observations, state, actions or rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.backend = env.backend.env if hasattr(env.backend, "env") else env.backend
        self.initial_z = {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def measurements(self):
        backend = self.backend
        gripper = backend.robots[0].gripper
        grip = np.asarray(backend.sim.data.get_site_xpos(gripper.important_sites["grip_site"]))
        objects = {}
        for name, obj in backend.objects_dict.items():
            position = np.asarray(backend.sim.data.body_xpos[backend.obj_body_id[name]])
            objects[name] = {
                "position_xyz": position.tolist(),
                "gripper_distance_m": float(np.linalg.norm(position-grip)),
                "both_fingerpads_contact": bool(backend._check_grasp(gripper, obj)),
                "lift_from_reset_m": float(position[2]-self.initial_z.get(name, position[2])),
            }
        return {"objects": objects, "gripper_xyz": grip.tolist(),
                "simulator_state": backend.sim.get_state().flatten().tolist(),
                "grasp_candidate_ids": [name for name, row in objects.items() if row["both_fingerpads_contact"]],
                "definition": "Native robosuite both-fingerpad contact; contact alone does not certify sustained grasp or target choice"}

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.initial_z = {name: float(self.backend.sim.data.body_xpos[body][2])
                          for name, body in self.backend.obj_body_id.items()}
        return observation, info | {"object_telemetry": self.measurements()}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info | {"object_telemetry": self.measurements()}

    def close(self):
        self.env.close()
