import numpy.random
import pandas as pd
from sympy.combinatorics import graycode

import xcs
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import pickle
import numpy.random
import moxcs
from moeadMethod import moeadMethod
from sympy.combinatorics.graycode import GrayCode

"""
    An implementation of an N-bit multiplexer problem for the X classifier system
"""

#The number of bits to use for the address in the multiplexer, 3 bit in example
bits = 1

#The maximum reward
rho = 1000

#The number of steps we learn for
learning_problems = 100

#The number of steps we validate for
validation_problems = 30

env = gym.make('MountainCar-v0')

#parameters
env._max_episode_steps = 1000

"""
    Returns a random state of the mountainCar
"""

pos_space = np.linspace(-1.2, 0.6, 12) #12  11
vel_space = np.linspace(-0.07, 0.07, 20) #20  19
action_space = [0, 1, 2]

# def state():
#     return ''.join(['0' if numpy.random.rand() > 0.5 else '1' for i in range(0, bits + 2**bits)])

interm = {'done':False}

import logging
# create logger with 'spam_application'
logger = logging.getLogger('moxcsnonGym')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('./data/moxcsnonGym.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

def getstate(observation):
    pos, vel =  observation
    pos_bin = int(np.digitize(pos, pos_space))
    vel_bin = int(np.digitize(vel, vel_space))

    return stateToCondition_gray(pos_bin,vel_bin)

def stateToCondition_binary(pos_bin,vel_bin):
    left = '{:05b}'.format(pos_bin)
    right = '{:05b}'.format(vel_bin)
    condition = left + right
    return condition

def stateToCondition_gray(pos_bin,vel_bin):
    left_gray = GrayCode(5)
    right_gray = GrayCode(5)
    left_binary = '{:05b}'.format(pos_bin)
    right_binary = '{:05b}'.format(vel_bin)
    left_gray = graycode.bin_to_gray(left_binary)
    right_gray = graycode.bin_to_gray(right_binary)
    condition = left_gray + right_gray
    return condition









"""
    solve mountain car problem, car move on the right
"""
def eop(done):
    return done

"""
    Calculates the reward for performing the action in the given state
"""
# def reward(state, action):
#     #Extract the parts from the state
#     address = state[:bits]
#     data = state[bits:]
#
#     #Check the action
#     if str(action) == data[int(address, 2)]:
#         return rho
#     else:
#         return 0
""
def initalize():
    done = False
    obs = env.reset()
    state = getstate(obs)
    return state
""

def reward(state, action):
    obs_next, reward, done, info = env.step(action)
    interm.done = done
    return reward, obs_next


def trainingTestingWeight_separate(weight):
    #training
    for j in range(learning_problems):
        # for each step
        # my_xcs.run_experiment()
        # my_xcs.run_experiment_seperate()
        logger.debug("iteration:%d, learning problems: %d"%(context['iteration'], j))
        my_xcs.run_iteration_episode(j, weight)

    # output
    #print("output*****action")
    my_xcs.stateAction(weight[0])

    # Testing
    this_correct = 0
    for j in range(validation_problems):
        # rand_state = getstate()
        # this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))
        #print("learning problems: ", j)

        if j<1:
            actionList=my_xcs.run_testing_episode(weight[0], j)
            dataframe = pd.DataFrame({'actionList': actionList})

            # 将DataFrame存储为csv,index表示是否显示行名，default=True
            filename = "actionList" + str(weight) + ".csv"
            dataframe.to_csv(filename, index=False, sep=',')
        else:
            my_xcs.run_testing_episode(weight[0], j)

    #print("Performance1 " + ": " + str((this_correct / validation_problems / rho) * 100) + "%");


def trainingTestingWeight(Allweight,ideaPoint, neighboursize):
    #training
    for j in range(learning_problems):
        # for each step
        # my_xcs.run_experiment()
        # my_xcs.run_experiment_seperate()
        logger.debug("iteration:%i, learning problems: %d"%(context['iteration'], j))
        my_xcs.run_iteration_episode(j, Allweight,ideaPoint, neighboursize)

    # output
    print("output*****action")
    for weight in Allweight:
        my_xcs.stateAction(weight)

    # Testing
    this_correct = 0
    for j in range(validation_problems):
        # rand_state = getstate()
        # this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))
        logger.debug("iteration:%d, learning problems: %d"%(context['iteration'], j))

        for weight in Allweight:
            if j<1:
                actionList=my_xcs.run_testing_episode(weight, j)
                dataframe = pd.DataFrame({'actionList': actionList})

                # 将DataFrame存储为csv,index表示是否显示行名，default=True
                filename = "iteration_%d_actionList_%s.csv"%(context['iteration'], str(weight))
                dataframe.to_csv(filename, index=False, sep=',')
            else:
                my_xcs.run_testing_episode(weight, j)

    #logger.debug("iteration:%d, Performance1: %s%"%(context['iteration'], str((this_correct / validation_problems / rho) * 100) ))



#Set some parameters
parameters = moxcs.parameter()
parameters.state_length = 10
parameters.p_hash=0
parameters.theta_mna = 3
parameters.e0 = 1000 * 0.01
parameters.theta_ga = 800000
parameters.gamma = 0.99
parameters.N = 8000000
parameters.beta =0.1
parameters.initial_error0=0.01
parameters.initial_fitness0=0.01
parameters.initial_prediction0=0.01
parameters.initial_error1=0.01
parameters.initial_fitness1=0.01
parameters.initial_prediction1=0.01


#Construct an XCS instance

context = {'iteration': 1, 'logger':logger}
my_xcs = moxcs.moxcs(parameters, getstate, reward, eop, initalize, stateToCondition_gray, context)


md = moeadMethod(2, 3, (-10, -10))
# 2 obj, 11weights
Allweights = md.initailize(2, 3)
#reAllweights=list(reversed(Allweights))
weights= [[[1, 0]],[[0, 1]]]
#weights= [[[1, 0]],[[0, 1]],[[0.9, 0.1]],[[0.1, 0.9]],[[0.8, 0.2]],[[0.2, 0.8]],[[0.3, 0.7]],[[0.7, 0.3]],[[0.6, 0.4]],[[0.4, 0.6]],[[0.5, 0.5]]]
#weights= [[1,0], [.5,.5],[0,1]]
#todo: 改正weight

for iteration in range(1, 3):
    context['iteration'] = iteration
    population = my_xcs.generate_population([[1,0]])
    trainingTestingWeight([[1,0]],[1000,1000], 1)

#method1 works separately
#training and testing by weight
# for weight in weights:
#     trainingTestingWeight_separate(weight)


#method2 doesnt work(run weights together), works when weights is [1,0],[0,1]
#trainingTestingWeight([[1,0],[0.9, 0.1],[0.8,0.2],[0.7,0.3],[0.6,0.4],[.5,.5],[0.4,0.6],[0.3,0.7],[0.2,0.8],[0.1,0.9],[0,1]],[1000,1000], 2)

#trainingTestingWeight([[0.9,0.1],[0.91,0.09],[0.92,0.08],[0.93,0.07],[0.94,0.06],[0.95,0.05],[0.94,0.06],[0.93,0.07],[0.94,0.06],[0.95,0.05],[0.96,0.04],[0.97,0.03],[0.98,0.02],[0.99,0.01],[0.9, 0.1]],[1000,1000], 1)










