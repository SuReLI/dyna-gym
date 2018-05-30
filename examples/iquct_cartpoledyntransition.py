import gym
import dyna_gym.envs.cartpole_dynamic_transition
import dyna_gym.agents.iquct as iquct

### Parameters
env = gym.make('CartPoleDynamicTransition-v0')
agent = iquct.IQUCT(
    action_space=env.action_space,
    gamma=0.9,
    rollouts=100,
    max_depth=1000,
    ucb_constant=0.7,
    regularization=1,
    degree=1
)
timesteps = 100
verbose = False

### Run
env.reset()
done = False
for ts in range(timesteps):
    __, __, done, __ = env.step(agent.act(env,done))
    if verbose:
        env.print_state()
    env.render()
    # Termination
    if ts+1 == timesteps:
        if verbose:
            print("Episode reached the end ({} timesteps)".format(ts+1))
    if done:
        if verbose:
            print("Episode finished after {} timesteps".format(ts+1))
        break
