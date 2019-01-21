import gym
import dyna_gym.envs.nscartpole_v2
import dyna_gym.agents.uct as uct

### Parameters
env = gym.make('NSCartPole-v2')
agent = uct.UCT(
    action_space=env.action_space,
    rollouts=100,
    horizon=50,
    is_model_dynamic=False
)

### Run
env.reset()
done = False
timeout = 1000
for ts in range(timeout):
    __, reward, done, __ = env.step(agent.act(env,done))
    env.render()
    if ts+1 == timeout:
        print("Successfully reached end of episode ({} timesteps)".format(ts+1))
    if done:
        print("Episode finished after {} timesteps".format(ts+1))
        break
