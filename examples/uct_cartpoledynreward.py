import gym
import dyna_gym.envs.cartpole_dynamic_transition
import dyna_gym.agents.uct as uct

### Parameters
env = gym.make('CartPoleDynamicReward-v0')
agent = uct.UCT(
    action_space=env.action_space,
    gamma=0.9,
    rollouts=100,
    max_depth=1000,
    is_model_dynamic=True
)
timesteps = 1000
verbose = True

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
