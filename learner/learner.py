import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import redis
import pickle as cPickle
import time
# 超参数
BATCH_SIZE = 32
LR = 0.01                   # learning rate
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target_net 更新频率
MEMORY_CAPACITY = 20000     # 一次采样数
ITER_COUNT = int(MEMORY_CAPACITY / BATCH_SIZE)
REDIS_SERVER = '192.168.100.8'  #为简化问题将redis部署在k8s-node1节点上
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


#TODO 回头再改成gpu版的
#use_cuda = torch.cuda.is_available()
#actor_evice = torch.device("cuda" if use_cuda else "cpu")
actor_device = torch.device("cpu")



class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class Learner(object):
    def __init__(self, host_ip):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.connect = redis.Redis(host=host_ip)                  # redis connect
        self.connect.delete('params')
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
    def get_enough_exp(self):
        #samples  = self.connect.get('experience')
        samples  = self.connect.lrange('experience', 0, -1)
        print (self.connect.llen('experience'))
        if samples is None:
            return False
        #if self.connect.llen('experience') < MEMORY_CAPACITY:
        if len(samples) < MEMORY_CAPACITY:
            return False
        #TODO sample转化为二维数组
        sample_list = []
        for sample in samples:
            sample_list.append(cPickle.loads(sample))
        self.memory = np.array(sample_list)
        print ("memory shape: ")
        print (self.memory.shape)
        self.connect.delete('experience')
        return True

    def learn(self):
        for iter_c in range(ITER_COUNT):
            # target parameter update
            if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
                self.target_net.load_state_dict(self.eval_net.state_dict())
                self.learn_step_counter = 0
            self.learn_step_counter += 1
            # sample batch transitions
            sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
            b_memory = self.memory[sample_index, :]
            b_s = torch.FloatTensor(b_memory[:, :N_STATES])
            b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
            b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
            b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

            # q_eval w.r.t the action in experience
            q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
            q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
            q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
            loss = self.loss_func(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        #一批样本学习完毕，往redis中更新参数 
        self.connect.set('params', cPickle.dumps(self.eval_net.to(actor_device).state_dict()))
        rn = np.random.randint(10000)
        self.connect.set('rn', cPickle.dumps(rn))
        print ("set params in redis success!") 
learner = Learner(REDIS_SERVER)
while True:
    if learner.get_enough_exp():
        print ("get enough experiences, begin to learn......")
        learner.learn()
    time.sleep(0.01) 

