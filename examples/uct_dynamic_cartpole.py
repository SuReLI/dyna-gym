import gym
import dyna_gym.envs.dynamic_cartpole_env
import dyna_gym.agents.uct as uct

env = gym.make('DynamicCartPole-v0')
env.reset()
agent = uct.UCT(
    action_space=env.action_space,
    gamma=0.9,
    rollouts=100,
    max_depth=1000,
    is_model_dynamic=False
)

timesteps = 100
done = False
#cumulative_theta = 0
for ts in range(timesteps):
    __, __, done, __ = env.step(agent.act(env,done))
    env.print_state()
    #cumulative_theta += abs(env.state[2])
    #env.render(close=True)
    #env.render()

    # Termination
    if ts+1 == timesteps:
        print("Episode reached the end ({} timesteps)".format(ts+1))
    if done:
        print("Episode finished after {} timesteps".format(ts+1))
        break

#print(cumulative_theta)
