import gym
import dyna_gym.envs.cartpole_dynamic_transition
import dyna_gym.agents.my_random_agent as ra

env = gym.make('CartPoleDynamicTransition-v0')
env.reset()
agent = ra.MyRandomAgent(env.action_space)

### Parameters
env = gym.make('CartPoleDynamicTransition-v0')
agent = ra.MyRandomAgent(env.action_space)
timesteps = 100
verbose = False

### Run
env.reset()
done = False
for ts in range(timesteps):
    __, __, done, __ = env.step(agent.act(0,0,0))
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
