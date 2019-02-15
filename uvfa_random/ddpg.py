#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import random

hidden1 = 300
hidden2 = 300
minibatch_size = 256
generator_learning_rate = 1e-4
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
gamma = 0.9
tau = 0.001
threshold = 0.01


# In[17]:


class Adam:
    def __init__(self, params, learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 1

    def train(self, gradient, params):
        self.m = self.beta1 * self.m + (1.0-self.beta1) * gradient
        self.v = self.beta2 * self.v + (1.0-self.beta2) * gradient**2
        m = self.m / (1.0-self.beta1**self.t)
        v = self.v / (1.0-self.beta2**self.t)
        self.t += 1
        return params - self.lr * m / (np.sqrt(v)+self.epsilon)
        #return params - self.lr * gradient


# In[18]:


class Dense:
    def __init__(self, weight, bias, learning_rate):
        self.weight = weight
        self.bias = bias
        #Adam Optimizer for weight and bias
        self.weight_optimizer = Adam(self.weight, learning_rate)
        self.bias_optimizer = Adam(self.bias, learning_rate)

    def set_input(self, inputs):
        self.inputs = inputs

    def outputs(self):
        return np.dot(self.inputs, self.weight) + self.bias

    #back propagation
    def gradient(self, diff):
        #chain rule
        input_gradient = np.dot(diff, self.weight.T)
        self.weight_gradient = np.dot(self.inputs.T, diff)
        self.bias_gradient = np.sum(diff, axis = 0)
        return input_gradient

    def train(self):
        self.weight = self.weight_optimizer.train(self.weight_gradient, self.weight)
        self.bias = self.bias_optimizer.train(self.bias_gradient, self.bias)


# In[19]:


class Relu:
    def _set(self, inputs):
        self.inputs = inputs
        zeros = np.zeros(self.inputs.shape)
        self.outputs = np.array([self.inputs, zeros]).max(axis = 0)

    #back propagation
    def gradient(self, diff):
        #dy/du
        dy_du = np.where(self.inputs < 0, 0, 1)
        #chain rule
        return diff * dy_du


# In[20]:


class Tanh:
    def _set(self, inputs):
        self.inputs = inputs
        self.outputs = np.tanh(self.inputs)

    #back propagation
    def gradient(self, diff):
        #dy/du
        dy_du = -1 * self.outputs**2 + 1
        #chain rule
        return diff * dy_du


# In[21]:


class Linear:
    def _set(self, inputs):
        self.inputs = inputs
        self.outputs = self.inputs

    #back propagation
    def gradient(self, diff):
        return diff


# In[22]:


class Layer:
    def __init__(self, weight, bias, learning_rate, activation_function):
        self.dense = Dense(weight, bias, learning_rate)
        if activation_function == "Relu":
            self.activation = Relu()
        elif activation_function == "Tanh":
            self.activation = Tanh()
        else:
            self.activation = Linear()

    def _set(self, inputs):
        self.dense.set_input(inputs)
        self.activation._set(self.dense.outputs())

    def outputs(self):
        return self.activation.outputs

    def gradient(self, diff):
        grad = self.activation.gradient(diff)
        return self.dense.gradient(grad)

    def train(self):
        self.dense.train()


# In[23]:


class Actor:
    def __init__(self, w1, b1, w2, b2, w3, b3, actor_learning_rate, action_high, action_low):
        self.layer1 = Layer(w1, b1, actor_learning_rate, "Relu")
        self.layer2 = Layer(w2, b2, actor_learning_rate, "Relu")
        self.layer3 = Layer(w3, b3, actor_learning_rate, "Tanh")
        self.action_high = action_high
        self.action_low = action_low

    def _set(self, state_batch):
        self.layer1._set(state_batch)
        self.layer2._set(self.layer1.outputs())
        self.layer3._set(self.layer2.outputs())

    def action(self, state_batch):
        self._set(state_batch)
        return self.layer3.outputs() * (self.action_high-self.action_low) / 2 + (self.action_high+self.action_low) / 2  #scaleout


# In[24]:


class ActorNetwork(Actor):
    def train(self, dq_da_batch):
        #gradient before adjustment
        grad = dq_da_batch * (self.action_high-self.action_low) / 2
        #back propagation
        for layer in [self.layer3, self.layer2, self.layer1]:
            grad = layer.gradient(grad)
            #policy gradient
            layer.dense.weight_gradient = -1 * layer.dense.weight_gradient / minibatch_size
            layer.dense.bias_gradient = -1 * layer.dense.bias_gradient / minibatch_size
            #train
            layer.train()

    def de_da_dg(self, de_da_batch, goal_size):
        grad = de_da_batch * (self.action_high-self.action_low) / 2
        #back propagation
        for layer in [self.layer3, self.layer2, self.layer1]:
            grad = layer.gradient(grad)
        return grad[[0: ,-1*goal_size:]


# In[25]:


class TargetActorNetwork(Actor):
    def update(self, actor_w1, actor_b1, actor_w2, actor_b2, actor_w3, actor_b3):
        self.layer1.dense.weight = actor_w1 * tau + self.layer1.dense.weight * (1.0-tau)
        self.layer1.dense.bias = actor_b1 * tau + self.layer1.dense.bias * (1.0-tau)
        self.layer2.dense.weight = actor_w2 * tau + self.layer2.dense.weight * (1.0-tau)
        self.layer2.dense.bias = actor_b2 * tau + self.layer2.dense.bias * (1.0-tau)
        self.layer3.dense.weight = actor_w3 * tau + self.layer3.dense.weight * (1.0-tau)
        self.layer3.dense.bias = actor_b3 * tau + self.layer3.dense.bias * (1.0-tau)


# In[26]:


class Critic:
    def __init__(self, w1, b1, w2, b2, w3, b3, critic_learning_rate):
        self.layer1 = Layer(w1, b1, critic_learning_rate, "Relu")
        self.layer2 = Layer(w2, b2, critic_learning_rate, "Relu")
        self.layer3 = Layer(w3, b3, critic_learning_rate, "Linear")

    def _set(self, state_batch, action_batch):
        inputs = np.concatenate([state_batch, action_batch], axis = 1)
        self.layer1._set(inputs)
        self.layer2._set(self.layer1.outputs())
        self.layer3._set(self.layer2.outputs())

    def qvalue(self, state_batch, action_batch):
        self._set(state_batch, action_batch)
        return self.layer3.outputs()


# In[27]:


class CriticNetwork(Critic):
    def train(self, qtarget, state_batch, action_batch):
        q = self.qvalue(state_batch, action_batch)
        #mean squared error
        loss = np.sum((qtarget-q)**2, axis = 0) / minibatch_size
        grad = 2 * (q-qtarget) / minibatch_size
        #back propagation
        for layer in [self.layer3, self.layer2, self.layer1]:
            grad = layer.gradient(grad)
            #train
            layer.train()
        return loss

    def dq_da(self, action_size, state_batch, action_batch):
        self._set(state_batch, action_batch)
        grad = np.ones((minibatch_size, 1))
        #back propagation
        for layer in [self.layer3, self.layer2, self.layer1]:
            grad = layer.gradient(grad)
        return grad[0: ,-1*action_size:]

    def de_dg(self, state_size, goal_size, inputs_batch, action_batch, qtarget):
        q = self.qvalue(inputs_batch, action_batch)
        loss = np.sum((qtarget-q)**2, axis = 0) / minibatch_size
        grad = -2 * (q-qtarget) / minibatch_size
        #back propagation
        for layer in [self.layer3, self.layer2, self.layer1]:
            grad = layer.gradient(grad)
        return grad[0: ,state_size:state_size+goal_size]



# In[30]:


class TargetCriticNetwork(Critic):
    def qtarget(self, next_state_batch, next_action_batch, reward_batch, done_batch):
        next_qvalue = self.qvalue(next_state_batch, next_action_batch)
        return reward_batch + gamma * next_qvalue * (1-done_batch)

    def update(self, critic_w1, critic_b1, critic_w2, critic_b2, critic_w3, critic_b3):
        self.layer1.dense.weight = critic_w1 * tau + self.layer1.dense.weight * (1.0-tau)
        self.layer1.dense.bias = critic_b1 * tau + self.layer1.dense.bias * (1.0-tau)
        self.layer2.dense.weight = critic_w2 * tau + self.layer2.dense.weight * (1.0-tau)
        self.layer2.dense.bias = critic_b2 * tau + self.layer2.dense.bias * (1.0-tau)
        self.layer3.dense.weight = critic_w3 * tau + self.layer3.dense.weight * (1.0-tau)
        self.layer3.dense.bias = critic_b3 * tau + self.layer3.dense.bias * (1.0-tau)


class Generator:
    def __init__(self, w1, b1, w2, b2, w3, b3, generator_learning_rate, goal_high, goal_low):
        self.layer1 = Layer(w1, b1, generator_learning_rate, "Relu")
        self.layer2 = Layer(w2, b2, generator_learning_rate, "Relu")
        self.layer3 = Layer(w3, b3, generator_learning_rate, "Tanh")
        self.goal_high = goal_high
        self.goal_low = goal_low

    def _set(self, state_batch):
        self.layer1._set(state_batch)
        self.layer2._set(self.layer1.outputs())
        self.layer3._set(self.layer2.outputs())

    def generate(self, state_batch):
        self._set(state_batch)
        outputs = self.layer3.outputs() * (self.goal_high-self.goal_low) / 2 + (self.goal_high+self.goal_low) / 2  #scaleout
        #convert goal



# In[24]:


class GeneratorNetwork(Generator):
    def train(self, de_dg_batch):
        #gradient before adjustment
        grad = de_dg_batch * (self.goal_high-self.goal_low) / 2
        #back propagation
        for layer in [self.layer3, self.layer2, self.layer1]:
            grad = layer.gradient(grad)
            #train
            layer.train()


# In[31]:


class Buffer:
    def __init__(self, buffer_size):
        self.transitions = []
        self.buffer_size = buffer_size

    def store(self, transition):
        self.transitions.append(transition)
        if len(self.transitions) > self.buffer_size:
            self.transitions = self.transitions[-1*self.buffer_size:]


# In[1]:


class Agent:
    def __init__(self, state_size, action_size, goal_size, action_high, action_low, goal_high, goal_low):
        #initialize actor
        w1 = np.random.randn(state_size+goal_size, hidden1) * np.sqrt(0.1/(state_size+goal_size))
        b1 = np.zeros(hidden1)
        w2 = np.random.randn(hidden1, hidden2) * np.sqrt(0.1/hidden1)
        b2 = np.zeros(hidden2)
        w3 = np.random.randn(hidden2, action_size) * np.sqrt(0.1/hidden2)
        b3 = np.zeros(action_size)
        self.actor_network = ActorNetwork(w1, b1, w2, b2, w3, b3, actor_learning_rate, action_high, action_low)
        self.target_actor_network = TargetActorNetwork(w1, b1, w2, b2, w3, b3, actor_learning_rate, action_high, action_low)
        #initialize critic
        w1 = np.random.randn(state_size+goal_size+action_size, hidden1) * np.sqrt(0.1/(state_size+goal_size+action_size))
        b1 = np.zeros(hidden1)
        w2 = np.random.randn(hidden1, hidden2) * np.sqrt(0.1/hidden1)
        b2 = np.zeros(hidden2)
        w3 = np.random.randn(hidden2, 1) * np.sqrt(0.1/hidden2)
        b3 = np.zeros(1)
        self.critic_network = CriticNetwork(w1, b1, w2, b2, w3, b3, critic_learning_rate)
        self.target_critic_network = TargetCriticNetwork(w1, b1, w2, b2, w3, b3, critic_learning_rate)
        #initialize generator
        w1 = np.random.randn(state_size, hidden1) * np.sqrt(0.1/state_size)
        b1 = np.zeros(hidden1)
        w2 = np.random.randn(hidden1, hidden2) * np.sqrt(0.1/hidden1)
        b2 = np.zeros(hidden2)
        w3 = np.random.randn(hidden2, goal_size) * np.sqrt(0.1/hidden2)
        b3 = np.zeros(goal_size)
        self.generator_network = GeneratorNetwork(w1, b1, w2, b2, w3, b3, generator_learning_rate, goal_high, goal_low)
        #set state information
        self.state_size = state_size
        #set action information
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        #set goal information
        self.goal_size = goal_size
        self.goal_high = goal_high
        self.goal_low = goal_low

    def choose_goal(self, state, restriction):
        goal = self.generator_network.generate(state)
        high_range = self.goal_high - goal
        low_range = self.goal_low - goal
        noise = np.random.uniform(low_range, high_range)
        noise = noise / restriction
        return goal + noise

    def choose_action(self, state, goal, restriction):
        #convert goal

        action = self.actor_network.action(np.concatenate([state, goal]))
        high_range = self.action_high - action
        low_range = self.action_low - action
        noise = np.random.uniform(low_range, high_range)
        noise = noise / restriction
        return action + noise

    def train(self, buffer):
        qloss = 0.0
        if minibatch_size <= len(buffer):
            #smaple a random minibatch
            minibatch = np.array(random.sample(buffer, minibatch_size))
            state_batch = np.vstack(minibatch[:,0])
            goal_batch = np.vstack(minibatch[:,1])
            action_batch = np.vstack(minibatch[:,2])
            next_state_batch = np.vstack(minibatch[:,3])
            reward_batch = np.vstack(minibatch[:,4])
            done_batch = np.vstack(minibatch[:,5])
            position_batch = np.vstack(minibatch[:,6])
            inputs_batch = np.concatenate([state_batch, goal_batch], axis = 1)
            next_inputs_batch = np.concatenate([next_state_batch, goal_batch], axis = 1)
            next_action_batch = self.target_actor_network.action(next_inputs_batch)
            #train critic network
            qtarget = self.target_critic_network.qtarget(next_inputs_batch, next_action_batch, reward_batch, done_batch)
            qloss = self.critic_network.train(qtarget, inputs_batch, action_batch)
            #train actor network
            dq_da_batch = self.critic_network.dq_da(self.action_size, inputs_batch, self.actor_network.action(inputs_batch))
            self.actor_network.train(dq_da_batch)
            #train generator network
            goal_batch = self.generator_network.generate(state_batch)
            reward_batch = np.vstack(-100*np.sqrt(np.sum((position_batch-goal_batch)**2, axis=1)))
            done_batch = np.vstack(np.where((np.sqrt(np.sum((position_batch-goal_batch)**2, axis=1))-threshold) > 0, 0, 1))
            inputs_batch = np.concatenate([state_batch, goal_batch], axis = 1)
            next_inputs_batch = np.concatenate([next_state_batch, goal_batch], axis = 1)
            next_action_batch = self.target_actor_network.action(next_inputs_batch)
            qtarget = self.target_critic_network.qtarget(next_inputs_batch, next_action_batch, reward_batch, done_batch)
            de_dg_batch = self.critic_network.de_dg(self.state_size, self.goal_size, inputs_batch, action_batch, qtarget)
            self.generator_network.train(de_dg_batch)
            #update target network
            self.target_critic_network.update(self.critic_network.layer1.dense.weight, self.critic_network.layer1.dense.bias,
                                              self.critic_network.layer2.dense.weight, self.critic_network.layer2.dense.bias,
                                              self.critic_network.layer3.dense.weight, self.critic_network.layer3.dense.bias)
            self.target_actor_network.update(self.actor_network.layer1.dense.weight, self.actor_network.layer1.dense.bias,
                                             self.actor_network.layer2.dense.weight, self.actor_network.layer2.dense.bias,
                                             self.actor_network.layer3.dense.weight, self.actor_network.layer3.dense.bias)
        return qloss

    def save(self, str):
        critic_params = [self.critic_network.layer1.dense.weight, self.critic_network.layer1.dense.bias,
                         self.critic_network.layer2.dense.weight, self.critic_network.layer2.dense.bias,
                         self.critic_network.layer3.dense.weight, self.critic_network.layer3.dense.bias]
        actor_params = [self.actor_network.layer1.dense.weight, self.actor_network.layer1.dense.bias,
                        self.actor_network.layer2.dense.weight, self.actor_network.layer2.dense.bias,
                        self.actor_network.layer3.dense.weight, self.actor_network.layer3.dense.bias]
        generator_params = [self.generator_network.layer1.dense.weight, self.generator_network.layer1.dense.bias,
                        self.generator_network.layer2.dense.weight, self.generator_network.layer2.dense.bias,
                        self.generator_network.layer3.dense.weight, self.generator_network.layer3.dense.bias]
        target_critic_params = [self.target_critic_network.layer1.dense.weight, self.target_critic_network.layer1.dense.bias,
                         self.target_critic_network.layer2.dense.weight, self.target_critic_network.layer2.dense.bias,
                         self.target_critic_network.layer3.dense.weight, self.target_critic_network.layer3.dense.bias]
        target_actor_params = [self.target_actor_network.layer1.dense.weight, self.target_actor_network.layer1.dense.bias,
                        self.target_actor_network.layer2.dense.weight, self.target_actor_network.layer2.dense.bias,
                        self.target_actor_network.layer3.dense.weight, self.target_actor_network.layer3.dense.bias]
        params = [critic_params, actor_params, generator_params, target_critic_params, target_actor_params]
        np.save(str, params)
