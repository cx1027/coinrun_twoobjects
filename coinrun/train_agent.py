"""
Train an agent using a PPO2 based on OpenAI Baselines.
"""

import time
from mpi4py import MPI
import random
from coinrun import setup_utils, make
import tensorflow as tf
from baselines.common import set_global_seeds
import coinrun.main_utils as utils
# from coinrun.import setup_utils, policies, wrappers, ppo2
from coinrun.config import Config

from MOXCS import moxcs
import csv
# from coinrun.Mask_RCNN.csci_e89_project import det

import numpy.random
import pandas as pd
# from sympy.combinatorics import graycode

# import xcs
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import pickle
import numpy.random
import MOXCS.XCSmountainGym
from MOXCS.moeadMethod import moeadMethod
# from sympy.combinatorics.graycode import GrayCode
# from getCondition import getCondition

"""
    An implementation of an N-bit multiplexer problem for the X classifier system
"""
num_envs=1
#The number of bits to use for the address in the multiplexer, 3 bit in example
bits = 1

#The maximum reward
rho = 1000

env_seed = '1'


#paper experiment
# learning_problems = 260
# validation_problems = 30
# interval = 20

#debug
learning_problems = 501
validation_problems = 5
interval = 10

nenvs=1
# arg_strs = setup_utils.setup_and_load()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# env = utils.make_general_env(nenvs, seed=5)


#parameters


"""
    Returns a random state of the mountainCar
"""

# pos_space = np.linspace(-1.2, 0.6, 12) #12  11
# vel_space = np.linspace(-0.07, 0.07, 20) #20  19
# action_space = [0, 1, 2]


#列行
maze1= [
[1,1],
[2,1],
[3,1],
[4,1],
[1,2],
[2,2],
[3,2],
[4,2],
[1,3],
[2,3],
[3,3],
[4,3],
[5,3],
[6,3],
[7,3],
[8,3],
[12,3],
[11,3],
[10,3],
[1,4],
[2,4],
[3,4],
[4,4],
[5,4],
[6,4],
[7,4],
[8,4],
[9,4],
[16,4],
[15,4],
[14,4],
[13,4],
[12,4],
[11,4],
[10,4],
[1,5],
[2,5],
[3,5],
[4,5],
[5,5],
[6,5],
[7,5],
[8,5],
[9,5],
[16,5],
[15,5],
[14,5],
[13,5],
[12,5],
[11,5],
[10,5],
[1,6],
[2,6],
[3,6],
[4,6],
[5,6],
[6,6],
[7,6],
[8,6],
[9,6],
[16,6],
[15,6],
[14,6],
[13,6],
[12,6],
[11,6],
[10,6],
[1,7],
[2,7],
[3,7],
[4,7],
[5,7],
[6,7],
[7,7],
[8,7],
[9,7],
[16,7],
[15,7],
[14,7],
[13,7],
[12,7],
[11,7],
[10,7],
[1,8],
[2,8],
[3,8],
[4,8],
[5,8],
[6,8],
[7,8],
[8,8],
[9,8],
[10,8],
[11,8],
[12,8],
[13,8],
[14,8],
[15,8],
[16,8]
]

maze2= [
[1,1],
[2,1],
[3,1],
[4,1],
[1,2],
[2,2],
[3,2],
[4,2],
[1,3],
[2,3],
[3,3],
[4,3],
[4,4],
[5,4],
[7,4],
[8,4],
[4,5],
[5,5],
[6,5],
[7,5],
[8,5],
[9,5]
]

# def state():
#     return ''.join(['0' if numpy.random.rand() > 0.5 else '1' for i in range(0, bits + 2**bits)])

interm = {'done':False}


# seed = 5
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
# set_global_seeds(seed)

# def getstate(observation):
#     pos, vel =  observation
#     pos_bin = int(np.digitize(pos, pos_space))
#     vel_bin = int(np.digitize(vel, vel_space))
#     return stateToCondition_gray(pos_bin,vel_bin)

def stateToCondition_binary(pos_bin,vel_bin):
    left = '{:05b}'.format(pos_bin)
    right = '{:05b}'.format(vel_bin)
    condition = left + right
    return condition

# def stateToCondition_gray(pos_bin,vel_bin):
    # left_gray = GrayCode(5)
    # right_gray = GrayCode(5)
    # left_binary = '{:05b}'.format(pos_bin)
    # right_binary = '{:05b}'.format(vel_bin)
    # left_gray = graycode.bin_to_gray(left_binary)
    # right_gray = graycode.bin_to_gray(right_binary)
    # condition = left_gray + right_gray
    # return condition

#todo: matrix(1024 1024 3) to coindition
# def stateToCondition(matrix):
#     left = getInformation(0)
#     right = getInformation(1)
#     up = getInformation(2)
#     down = getInformation(3)
#     return left+right+up+down

#todo:get information from matrix
# class InferenceConfig(det.DetConfig):
#     # Set batch size to 1 since we'll be running inference on
#     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

# inf_config = InferenceConfig('obj', ['obj','player_obj', 'monster_obj', 'step_obj', 'wall_obj'])

def stateToCodution(conditionChar):
    #maze1 A:001 S:010 .:100 $:101 %:110 1:111, T:011, N:000, need all 8
    #maze2 A:001 S:010 .:100 $:101 %:110 1:111, T:011, #:000, need STA.#1,
    #TODO: # TO N
    # maze2 A:001 S:010 .:100 $:101 %:110 1:111, T:011, N:000, need STA.N1
    # print("conditionChar[i]")
    # print(conditionChar)
    condition = list()
    for i in range(len(conditionChar)):
        # print(conditionChar[i])
        if conditionChar[i].decode()=='A':
            condition.append('001')
        elif conditionChar[i].decode()=='S':
            condition.append('010')
        elif conditionChar[i].decode()=='.':
            condition.append('100')
        elif conditionChar[i].decode() == '$':
            condition.append('101')
        elif conditionChar[i].decode()=='%':
            condition.append('110')
        elif conditionChar[i].decode() == 'T':
            condition.append('011')
        elif conditionChar[i].decode()=='N':
            condition.append('000')
        else:
            condition.append('111')
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
#     #Check the action
#     if str(action) == data[int(address, 2)]:
#         return rho
#     else:
#         return 0
""
def initalize():
    done = False
    firstCondition = env.getFirstCondition()
    state = stateToCodution(firstCondition)
    return state
""

def reward(state, action):
    obs_next, reward, done, info, condition = env.step(action)
    interm.done = done
    return reward, obs_next, condition


def trainingTestingWeight_separate(weight):
    # set_global_seeds(seed)
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
    # dataframe=[]
    for j in range(validation_problems):
        # rand_state = getstate()
        # this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))
        #print("learning problems: ", j)
        print("start to test")
        print(j)

        # if j<1:
        resultList=my_xcs.run_testing_episode(weight, j)
        dataframe.append(resultList)
        dataframe = pd.DataFrame({'validation problem': j, 'result': resultList})

    print("dataframe")
    print(dataframe)

            # 将DataFrame存储为csv,index表示是否显示行名，default=True
    filename = "resultList.csv"
    dataframe.to_csv(filename, index=True, sep=',')
        # else:
        #     my_xcs.run_testing_episode(weight[0], j)

    #print("Performance1 " + ": " + str((this_correct / validation_problems / rho) * 100) + "%");

#backup
# currx_training=14
# currx_testing=4
# non_markov_block_training=18
# non_markov_block_testing=3

currx_training=5
currx_testing=5
non_markov_block_training=4
non_markov_block_testing=4

def trainingTestingWeight(Allweight,ideaPoint, neighboursize, iteration, TestingInterval):

    #training
    my_xcs.emptyPopulation()

    setup_utils.setup_and_load(environment=env_seed)
    env = make('standard', num_envs=num_envs, curr_x=currx_training, non_markov_block=non_markov_block_training)
    env._max_episode_steps = 1000
    nenvs = Config.NUM_ENVS
    total_timesteps = int(256e6)

    df = pd.DataFrame(columns=['currentweight', 'result', 'stays', 'validationproblem','inteval', 'reward'])
    for j in range(learning_problems):
        setup_utils.setup_and_load(environment=env_seed)
        logger.debug("iteration:%i, learning problems: %d"%(context['iteration'], j))
        #env1=--run-id myrun --num-levels 1 --set-seed 1
        my_xcs.run_iteration_episode(j, Allweight,ideaPoint, neighboursize)
        print("testing j:", j)

    # output
    # print("output population")
    # #todo:print later
    # #for weight in Allweight:
    # print(len(my_xcs.population))


        if j>1 and (j+1)%TestingInterval==0:
            print("testing: ",j)
            # my_xcs.print_population()
        # Testing

            setup_utils.setup_and_load(environment='1')
            nenvs = Config.NUM_ENVS

            env = make('standard', num_envs=num_envs, curr_x=currx_testing,non_markov_block=non_markov_block_testing)
            env._max_episode_steps = 1000


            resultAllweight=[]
            validationproblem=[]
            currentweight=[]
            stays=[]
            inteval=[]
            rewList=[]
            for i in range(validation_problems):
                # rand_state = getstate()
                # this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))
                logger.debug("iteration:%d, learning problems: %d"%(context['iteration'], j))

                for weight in Allweight:
                    # if j<1:
                    #     actionList=my_xcs.run_testing_episode(weight, j)
                    #     dataframe = pd.DataFrame({'actionList': actionList})
                    #
                    #     # 将DataFrame存储为csv,index表示是否显示行名，default=True
                    #     filename = "iteration_%d_actionList_%s.csv"%(context['iteration'], str(weight))
                    #     dataframe.to_csv(filename, index=False, sep=',')
                    # else:
                    #     my_xcs.run_testing_episode(weight, j)

                    # env2=--run-id myrun --num-levels 1 --set-seed 5
                    resultList, stay, rew = my_xcs.run_testing_episode(weight, i, iteration)
                    resultAllweight.append(resultList)
                    validationproblem.append(i)
                    currentweight.append(weight)
                    stays.append(stay)
                    inteval.append(j)
                    rewList.append(rew)
                    # df=df.append(pd.DataFrame({'currentweight': weight, 'result': resultList,'stays':stay,'validationproblem':i,'inteval':inteval}),ignore_index=True)


            dataframe = pd.DataFrame({'currentweight': currentweight, 'result': resultAllweight,'stays':stays,'validationproblem':i,'inteval':j,'reward':rewList})
            df=df.append(dataframe, ignore_index=True)



    filename = "result_" + "interation_" +str(iteration) + ".csv"
    df.to_csv(filename, index=False, sep=',')

    # return dataframe


    #logger.debug("iteration:%d, Performance1: %s%"%(context['iteration'], str((this_correct / validation_problems / rho) * 100) ))


setup_utils.setup_and_load(environment =env_seed)
# set_global_seeds(seed)
env = make('standard', num_envs=num_envs, curr_x=currx_training, non_markov_block=non_markov_block_training)
env._max_episode_steps = 1000

nenvs = Config.NUM_ENVS
total_timesteps = int(256e6)
# save_interval = args.save_interval



    # Set some parameters
parameters = moxcs.parameter()
# parameters.state_length = 10
parameters.state_length = 24
parameters.num_actions =7
parameters.p_hash = 0.01

parameters.theta_mna = 7
parameters.e0 = 1000 * 0.01
parameters.theta_ga = 800000
parameters.gamma = 0.99
# parameters.gamma = 0.99
parameters.N = 8000000
parameters.beta = 0.1
parameters.initial_error0 = 0.01
parameters.initial_fitness0 = 0.01
parameters.initial_prediction0 = 0.0
parameters.initial_error1 = 0.01
parameters.initial_fitness1 = 0.01
parameters.initial_prediction1 = 0.0
#todo: new parameters
parameters.state_length = 24  # The number of bits in the state
parameters.num_actions=7
parameters.bitRange = 3
parameters.bits = 24

    # Construct an XCS instance

context = {'iteration': 1, 'logger': logger}
my_xcs = moxcs.moxcs(parameters, stateToCodution, reward, eop, initalize, context, env)

md = moeadMethod(2, 3, (-10, -10))
    # 2 obj, 11weights
Allweights = md.initailize(2, 3)
    # reAllweights=list(reversed(Allweights))
weights = [[[1, 0]], [[0, 1]]]




def main():
    positions = maze2
    print("aaaaaaaaaaa")
    # result=[]
    for iteration in range(0,30):
    # context['iteration'] = iteration
        #todo:population is wrong with paraters.state_length
        #population = my_xcs.generate_population([[1, 0]],positions)
        population = my_xcs.allHashClassifier([[1, 0]])#weights, bits:多少位, range:每一位取值
        # print("population")

        trainingTestingWeight([[1,0],[0,1]], [1000, 1000], 1, iteration,interval)

    # with open('resultlist', 'w', newline='') as myfile:
    #         wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #         wr.writerow(result)





if __name__ == '__main__':
    main()
