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

### Run
env.reset()
done = False
verbose = False
timeout = 1000
for ts in range(timeout):
    __, reward, done, __ = env.step(agent.act(env,done))
    if verbose:
        env.print_state()
    env.render()
    if ts+1 == timeout:
        print("Successfully reached end of episode ({} timeout)".format(ts+1))
    if done:
        print("Episode finished after {} timeout".format(ts+1))
        break
