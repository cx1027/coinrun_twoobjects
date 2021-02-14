# import numpy.random
# import pandas as pd
# # from sympy.combinatorics import graycode
#
# import xcs
# import gym
# import matplotlib.pyplot as plt
# import numpy as np
# from gym import wrappers
# import pickle
# import numpy.random
# import coinrun.MOXCS.moxcs as moxcs
# from coinrun.MOXCS.moeadMethod import moeadMethod
# # from sympy.combinatorics.graycode import GrayCode
# import coinrun.coinrun.main_utils as utils
# from coinrun.coinrun import setup_utils, policies, wrappers, ppo2
# from mpi4py import MPI
#
# from coinrun.coinrun.train_agent import stateToCodution
#
# """
#     An implementation of an N-bit multiplexer problem for the X classifier system
# """
#
# #The number of bits to use for the address in the multiplexer, 3 bit in example
# bits = 1
#
# #The maximum reward
# rho = 1000
#
# #The number of steps we learn for
# learning_problems = 100
#
# #The number of steps we validate for
# validation_problems = 3
# nenvs=1
# args = setup_utils.setup_and_load()
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
#
# env = utils.make_general_env(nenvs, seed=rank)
#
# #parameters
# env._max_episode_steps = 1000
#
# """
#     Returns a random state of the mountainCar
# """
#
# pos_space = np.linspace(-1.2, 0.6, 12) #12  11
# vel_space = np.linspace(-0.07, 0.07, 20) #20  19
# action_space = [0, 1, 2]
#
# # def state():
# #     return ''.join(['0' if numpy.random.rand() > 0.5 else '1' for i in range(0, bits + 2**bits)])
#
# interm = {'done':False}
#
# # def getstate(observation):
# #     pos, vel =  observation
# #     pos_bin = int(np.digitize(pos, pos_space))
# #     vel_bin = int(np.digitize(vel, vel_space))
# #     return stateToCondition_gray(pos_bin,vel_bin)
#
# def stateToCondition_binary(pos_bin,vel_bin):
#     left = '{:05b}'.format(pos_bin)
#     right = '{:05b}'.format(vel_bin)
#     condition = left + right
#     return condition
#
# # def stateToCondition_gray(pos_bin,vel_bin):
#     # left_gray = GrayCode(5)
#     # right_gray = GrayCode(5)
#     # left_binary = '{:05b}'.format(pos_bin)
#     # right_binary = '{:05b}'.format(vel_bin)
#     # left_gray = graycode.bin_to_gray(left_binary)
#     # right_gray = graycode.bin_to_gray(right_binary)
#     # condition = left_gray + right_gray
#     # return condition
#
# #todo: update range should be -32,+32
# xais = np.linspace(0, 64, 16) #16个bucket
#
# #todo: get observation from mask-rcnn
# #todo:condition is float[10], state is int[10], each int is between 1 to 16 (0-64 16个区间,0 means not match)
#
# def integerInput10(observation):
#     integerList = []
#     for i in range(10):
#         integerList[i] = int(np.digitize(observation[i], xais))
#     return integerList
#
#
#
#
#
#
#
#
# """
#     solve mountain car problem, car move on the right
# """
# def eop(done):
#     return done
#
# """
#     Calculates the reward for performing the action in the given state
# """
# # def reward(state, action):
# #     #Extract the parts from the state
# #     address = state[:bits]
# #     data = state[bits:]
# #
# #     #Check the action
# #     if str(action) == data[int(address, 2)]:
# #         return rho
# #     else:
# #         return 0
# ""
# def initalize():
#     done = False
#     obs = env.reset()
#     # state = getstate(obs)
#     state=stateToCodution(obs)
#     return state
# ""
#
# def reward(state, action):
#     obs_next, reward, done, info = env.step(action)
#     interm.done = done
#     return reward, obs_next
#
#
# def trainingTestingWeight_separate(weight):
#     #training
#     for j in range(learning_problems):
#         # for each step
#         # my_xcs.run_experiment()
#         # my_xcs.run_experiment_seperate()
#         print("learning problems: ", j)
#         my_xcs.run_iteration_episode(j, weight)
#
#     # output
#     #print("output*****action")
#     my_xcs.stateAction(weight[0])
#
#     # Testing
#     this_correct = 0
#     for j in range(validation_problems):
#         # rand_state = getstate()
#         # this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))
#         #print("learning problems: ", j)
#
#         if j<1:
#             actionList=my_xcs.run_testing_episode(weight[0], j)
#             dataframe = pd.DataFrame({'actionList': actionList})
#
#             # 将DataFrame存储为csv,index表示是否显示行名，default=True
#             filename = "actionList" + str(weight) + ".csv"
#             dataframe.to_csv(filename, index=False, sep=',')
#         else:
#             my_xcs.run_testing_episode(weight[0], j)
#
#     #print("Performance1 " + ": " + str((this_correct / validation_problems / rho) * 100) + "%");
#
#
# def trainingTestingWeight(Allweight,ideaPoint, neighboursize):
#     #training
#     for j in range(learning_problems):
#         # for each step
#         # my_xcs.run_experiment()
#         # my_xcs.run_experiment_seperate()
#         print("learning problems: ", j)
#         my_xcs.run_iteration_episode(j, Allweight,ideaPoint, neighboursize)
#
#     # output
#     print("output*****action")
#     for weight in Allweight:
#         my_xcs.stateAction(weight)
#
#     # Testing
#     this_correct = 0
#     for j in range(validation_problems):
#         # rand_state = getstate()
#         # this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))
#         print("learning problems: ", j)
#
#         for weight in Allweight:
#             if j<1:
#                 actionList=my_xcs.run_testing_episode(weight, j)
#                 dataframe = pd.DataFrame({'actionList': actionList})
#
#                 # 将DataFrame存储为csv,index表示是否显示行名，default=True
#                 filename = "actionList" + str(weight) + ".csv"
#                 dataframe.to_csv(filename, index=False, sep=',')
#             else:
#                 my_xcs.run_testing_episode(weight, j)
#
#     print("Performance1 " + ": " + str((this_correct / validation_problems / rho) * 100) + "%");
#
#
#
# #Set some parameters
# parameters = moxcs.parameter()
# parameters.state_length = 10
# parameters.p_hash=0
# parameters.theta_mna = 3
# parameters.e0 = 1000 * 0.01
# parameters.theta_ga = 800000
# parameters.gamma = 0.99
# parameters.N = 40000000000000000000
# parameters.beta =0.1
# parameters.initial_error0=0.01
# parameters.initial_fitness0=0.01
# parameters.initial_prediction0=0.01
# parameters.initial_error1=0.01
# parameters.initial_fitness1=0.01
# parameters.initial_prediction1=0.01
#
#
# #Construct an XCS instance
# my_xcs = moxcs.moxcs(parameters, integerInput10, reward, eop, initalize, integerInput10)
#
#
# md = moeadMethod(2, 3, (-10, -10))
# # 2 obj, 11weights
# Allweights = md.initailize(2, 3)
# #reAllweights=list(reversed(Allweights))
# weights= [[[1, 0]],[[0, 1]]]
# #weights= [[[1, 0]],[[0, 1]],[[0.9, 0.1]],[[0.1, 0.9]],[[0.8, 0.2]],[[0.2, 0.8]],[[0.3, 0.7]],[[0.7, 0.3]],[[0.6, 0.4]],[[0.4, 0.6]],[[0.5, 0.5]]]
# #weights= [[1,0], [.5,.5],[0,1]]
# #todo: 改正weight
# population = my_xcs.generate_population([[.99,.01]],10,16)
# print(len(population))
#
# #method1 works separately
# #training and testing by weight
# # for weight in weights:
# #     trainingTestingWeight_separate(weight)
#
#
# #method2 doesnt work(run weights together), works when weights is [1,0],[0,1]
# #trainingTestingWeight([[1,0],[0.9, 0.1],[0.8,0.2],[0.7,0.3],[0.6,0.4],[.5,.5],[0.4,0.6],[0.3,0.7],[0.2,0.8],[0.1,0.9],[0,1]],[1000,1000], 2)
#
# #trainingTestingWeight([[0.9,0.1],[0.91,0.09],[0.92,0.08],[0.93,0.07],[0.94,0.06],[0.95,0.05],[0.94,0.06],[0.93,0.07],[0.94,0.06],[0.95,0.05],[0.96,0.04],[0.97,0.03],[0.98,0.02],[0.99,0.01],[0.9, 0.1]],[1000,1000], 1)
#
# trainingTestingWeight([[.99,.01]],[1000,1000], 1)
#
#
#
#
#
#
#
#
#
