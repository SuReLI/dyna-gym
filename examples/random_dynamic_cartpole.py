import gym
import dyna_gym.agents.my_random_agent as ra
import dyna_gym.envs.dynamic_cartpole_env

env = gym.make('DynamicCartPole-v0')
env.reset()
agent = ra.MyRandomAgent(env.action_space)

for t in range(1000):
    env.render()
    env.step(agent.act(0,0,0))
