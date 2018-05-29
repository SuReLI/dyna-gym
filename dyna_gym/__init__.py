from gym.envs.registration import register

register(
    id='CartPoleDynamicTransition-v0',
    entry_point='dyna_gym.envs:CartPoleDynamicTransition',
)

register(
    id='CartPoleDynamicReward-v0',
    entry_point='dyna_gym.envs:CartPoleDynamicReward',
)
