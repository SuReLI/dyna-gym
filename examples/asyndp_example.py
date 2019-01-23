import gym
import numpy as np
import dyna_gym.agents.my_random_agent as ra
import dyna_gym.agents.asynchronous_dp as asyndp

#np.random.seed(1993)

### Parameters
#env = gym.make('RandomNSMDP-v0')
#env = gym.make('NSFrozenLake-v0')
#env = gym.make('NSCliff-v0')
env = gym.make('NSBridge-v0')

#agent = ra.MyRandomAgent(env.action_space)
agent = asyndp.AsynDP(env.action_space, gamma=0.9, max_depth=4, is_model_dynamic=False)

agent.display()

### Run
done = False
env.render()
timeout = 20
undiscounted_return, total_time, discounted_return = 0.0, 0, 0.0
for t in range(timeout):
    action = agent.act(env,done)
    _, reward, done, _ = env.step(action)
    undiscounted_return += reward
    discounted_return += (agent.gamma**t) * reward
    print()
    env.render()
    if (t + 1 == timeout) or done:
        total_time = t + 1
        break
print('End of episode')
print('Total time          :', total_time)
print('Discounted return   :', discounted_return)
print('Undiscounted return :', undiscounted_return)
