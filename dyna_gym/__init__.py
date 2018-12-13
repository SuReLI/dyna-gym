from gym.envs.registration import register

register(
    id='NSChain-v0',
    entry_point='dyna_gym.envs:NSChain',
)

register(
    id='RandomNSMDP-v0',
    entry_point='dyna_gym.envs:RandomNSMDP',
)

register(
    id='NSFrozenLakeEnv-v0',
    entry_point='dyna_gym.envs:NSFrozenLakeEnv',
)

register(
    id='NSCartPole-v0',
    entry_point='dyna_gym.envs:NSCartPoleV0',
)

register(
    id='NSCartPole-v1',
    entry_point='dyna_gym.envs:NSCartPoleV1',
)

register(
    id='NSCartPole-v2',
    entry_point='dyna_gym.envs:NSCartPoleV2',
)
