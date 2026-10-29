"""Worker-local registration; existing campaigns and upstream files stay unchanged."""
def register():
    from multibench.torch_model import install as model
    from multibench.rl_env import install as environment
    model()
    environment()
