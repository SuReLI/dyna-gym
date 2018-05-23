from gym.envs.registration import register

register(
    id='DynamicCartPole-v0',
    entry_point='dyna_gym.envs:DynamicCartPole',
)
