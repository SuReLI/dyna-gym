import gym
import dyna_gym.envs.dynamic_cartpole_env
import dyna_gym.agents.oluct as oluct

env = gym.make('DynamicCartPole-v0')
env.reset()
agent = oluct.OLUCT(action_space=env.action_space, gamma=0.9, rollouts=100, max_depth=1000)

timesteps = 100
done = False
for t in range(timesteps):
    __, __, done, __ = env.step(agent.act(env,done))
    env.print_state()
    #env.render(close=True)
    #env.render()

    # Termination
    if t+1 == timesteps:
        print("Episode finished after {} timesteps (maximum timesteps)".format(t+1))
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
