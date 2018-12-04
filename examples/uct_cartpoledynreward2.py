import gym
import dyna_gym.envs.cartpole_dynamic_transition
import dyna_gym.agents.uct as uct

### Parameters
env = gym.make('CartPoleDynamicReward-v2')
agent = uct.UCT(
    action_space=env.action_space,
    rollouts=100,
    horizon=50,
    is_model_dynamic=True
)
timesteps = 1000
verbose = False

### Run
env.reset()
done = False
for ts in range(timesteps):
    __, reward, done, __ = env.step(agent.act(env,done))
    if verbose:
        env.print_state()
    env.render()
    if ts+1 == timesteps:
        print("Successfully reached end of episode ({} timesteps)".format(ts+1))
    if done:
        print("Episode finished after {} timesteps".format(ts+1))
        break
