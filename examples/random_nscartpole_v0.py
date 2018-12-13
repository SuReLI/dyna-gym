import gym
import dyna_gym.envs.nscartpole_v0
import dyna_gym.agents.my_random_agent as ra

### Parameters
env = gym.make('NSCartPole-v0')
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
    if ts+1 == timesteps:
        print("Successfully reached end of episode ({} timesteps)".format(ts+1))
    if done:
        print("Episode finished after {} timesteps".format(ts+1))
        break
