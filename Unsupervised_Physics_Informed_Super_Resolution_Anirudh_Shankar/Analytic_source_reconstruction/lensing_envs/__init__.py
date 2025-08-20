from gymnasium.envs.registration import register

register(
    id="Source-v0",
    entry_point="lensing_envs.lensing_envs:Source",
)