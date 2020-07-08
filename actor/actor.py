import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import gym
import redis
import pickle as cPickle

EPSILON = 0.9               # 贪婪策略
MEMORY_CAPACITY = 20000
REDIS_SERVER = '192.168.100.8' #为简化问题将redis部署在k8s-node1节点上  
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # 初始化 
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # 初始化

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class Actor(object):
    def __init__(self, host_ip):
        self.policy_net =  Net()
        self.connect = redis.Redis(host=host_ip) #default port 6379 
        self.connect.delete('experience') #初始化时删除历史采样数据
        self.rn = 1   #learner每更新一次网络参数会使用随机数(random number)更新redis中'rn'的值

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.policy_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        exp = cPickle.dumps(transition)
        #sample  = self.connect.get('experience')
        sample  = self.connect.lrange('experience', 0, -1)
        #第一次往里插入数据key值不存在
        if sample is None:
            self.connect.rpush('experience', exp)#
            return
        num_samples = self.connect.llen('experience') 
        if num_samples % 1000 == 0:
            print("current time: ",time.strftime('%Y.%m.%d %H:%M:%S ',time.localtime(time.time())))
            print (" num_samples: " + str(num_samples))
        if num_samples < MEMORY_CAPACITY :
            self.connect.rpush('experience', exp)
        else:
            #样本已经采够了
            time.sleep(0.01) #等待learner取完数据
        

    def update_params(self):
        params = self.connect.get('params')
        if not params is None:
            print("Sync params....")
            self.policy_net.load_state_dict(cPickle.loads(params))

    #保证actor取到最新的params
    def is_newest_para(self):
        t = self.connect.get('rn')
        if t is None:
            return False
        if self.rn == cPickle.loads(t):
            return False
        else:
            self.rn = cPickle.loads(t)
            return True



actor = Actor(REDIS_SERVER)

print('\nCollecting experience...')
while True:
    s = env.reset()
    episod_reward = 0
    while True:
        if actor.is_newest_para():
            actor.update_params()
        a = actor.choose_action(s)
        # take action
        s_, r, done, info = env.step(a)
        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        actor.store_transition(s, a, r, s_)
        episod_reward += r
        #杆子倒了一个episode结束
        if done:
            print('Episode_reward: ', round(episod_reward, 2))
            break
        s = s_
