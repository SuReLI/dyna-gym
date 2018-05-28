import gym
import dyna_gym.envs.dynamic_cartpole_env
import dyna_gym.agents.iquct as iquct

### Parameters
env = gym.make('DynamicCartPole-v0')
agent = iquct.IQUCT(
    action_space=env.action_space,
    gamma=0.9,
    rollouts=100,
    max_depth=1000,
    is_model_dynamic=False
)
timesteps = 100
verbose = True

### Run
env.reset()
done = False
cumulative_theta = 0 #TRM
for ts in range(timesteps):
    __, __, done, __ = env.step(agent.act(env,done))
    if verbose:
        env.print_state()
    cumulative_theta += abs(env.state[2]) #TRM
    # Termination
    if ts+1 == timesteps:
        if verbose:
            print("Episode reached the end ({} timesteps)".format(ts+1))
    if done:
        if verbose:
            print("Episode finished after {} timesteps".format(ts+1))
        break
print(cumulative_theta)
