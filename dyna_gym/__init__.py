from gym.envs.registration import register

register(
    id='NSChain-v0',
    entry_point='dyna_gym.envs:NSChain',
)

register(
    id='CartPoleDynamicTransition-v0',
    entry_point='dyna_gym.envs:CartPoleDynamicTransition',
)

register(
    id='CartPoleDynamicReward-v1',
    entry_point='dyna_gym.envs:CartPoleDynamicRewardV1',
)

register(
    id='CartPoleDynamicReward-v2',
    entry_point='dyna_gym.envs:CartPoleDynamicRewardV2',
)
