import gym
import dyna_gym.envs.cartpole_dynamic_transition
import dyna_gym.agents.mcts as mcts

### Parameters
env = gym.make('CartPoleDynamicTransition-v0')
agent = mcts.MCTS(
    action_space=env.action_space,
    gamma=0.9,
    rollouts=100,
    max_depth=1000,
    is_model_dynamic=True
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
    if ts+1 == timesteps:
        print("Successfully reached end of episode ({} timesteps)".format(ts+1))
    if done:
        print("Episode finished after {} timesteps".format(ts+1))
        break
