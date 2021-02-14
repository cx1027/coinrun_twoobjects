import numpy as np
import numpy.random
import itertools
from copy import deepcopy
import gym
from MOXCS.moeadMethod import moeadMethod
import pandas as pd
import random
import csv
from coinrun.coinrunenv import CoinRunVecEnv



"""
A class that represents the parameters of an XCS system
"""

step = 0


class parameter:
    def __init__(self):
        self.state_length = 24  # The number of bits in the state
        self.num_actions = 3  # The number of actions in this system
        self.theta_mna = 3  # The minimum number of elements in the match set
        self.initial_prediction0 = 0.01  # The initial prediction value in classifiers
        self.initial_error0 = 0.01  # The initial error value in classifiers
        self.initial_fitness0 = 0.01  # The initial fitness value in classifiers
        self.initial_prediction1 = 0.01  # The initial prediction value in classifiers
        self.initial_error1 = 0.01  # The initial error value in classifiers
        self.initial_fitness1 = 0.01  # The initial fitness value in classifiers
        self.p_hash = 0.3  # The probability of generating a hash in a condition
        self.prob_exploration = 0.5  # The probability of the system exploring the environment
        self.gamma = 0.71  # The payoff decay rate
        self.alpha = 0.1
        self.beta = 0.2
        self.nu = 5  #
        self.N = 400  # The maximum number of classifiers in the population
        self.e0 = 0.01  # The minimum error value
        self.theta_del = 25  # The experience level below which we don't delete classifiers
        self.delta = 0.1  # The multiplier for the deletion vote of a classifier
        self.theta_sub = 20  # The rate of subsumption
        self.theta_ga = 25  # The rate of the genetic algorithm
        self.crossover_rate = 0.8
        self.mutation_rate = 0.04
        self.do_GA_subsumption = True
        self.do_action_set_subsumption = True
        self.bitRange = 16
        self.bits = 24


"""
A classifier in the X Classifier System
"""
def to_str(var):
    if type(var) is list:
        return str(var)[1:-1] # list
    if type(var) is np.ndarray:
        try:
            return str(list(var[0]))[1:-1] # numpy 1D array
        except TypeError:
            return str(list(var))[1:-1] # numpy sequence
    return str(var)

class classifier:
    global_id = 0  # A Globally unique identifier

    def __init__(self, parameter, state=None, weight=[], position = [0,0]):
        self.id = classifier.global_id
        classifier.global_id = classifier.global_id + 1
        self.action = numpy.random.randint(0, parameter.num_actions)
        self.prediction = np.array([parameter.initial_prediction0, parameter.initial_prediction1])
        self.error = np.array([parameter.initial_error0, parameter.initial_error1])
        self.fitness = np.array([parameter.initial_fitness0, parameter.initial_fitness1])
        self.fitnessAvg = parameter.initial_fitness0
        self.experience = 0
        self.time_stamp = 0
        self.average_size = 1
        self.numerosity = 1
        self.weight = weight
        self.position = position
        # self.coin_env = CoinRunVecEnv('a','a')

        #todo: !!!!!!!!!!!!!condition
        if state==None:
            self.condition = ''.join(
                ['#' if numpy.random.rand() < parameter.p_hash else '0' if numpy.random.rand() > 0.5 else '1' for i in
                 [0] * parameter.state_length])
        else:
            # print("info:")
            # print(type(state))

            state = ''.join(list(state))
            # print(parameter.state_length)
            self.condition = ''.join(
                ['#' if numpy.random.rand() < parameter.p_hash else state[i] for i in range(parameter.state_length)])

        # Generate the condition from the state (if supplied)
        # join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串
        #todo:定义integer input
        # self.condition = ''.join(
        #     ['#' if numpy.random.rand() < parameter.p_hash else state[i] for i in range(parameter.state_length)])
        # self.condition = np.random.randint(0,parameter.bitRange,parameter.bits)

        # self.left = 0
        # self.right = 0

    def __str__(self):
        return "Classifier " + str(self.id) + ": " + self.condition + " = " + str(self.action) + " fitness[0]: " + str(
            self.fitness[0]) + " prediction[0]: " + str(self.prediction[0]) + " error[0]: " + str(
            self.error[0]) + " fitness[1]: " + str(
            self.fitness[1]) + " prediction[1]: " + str(self.prediction[1]) + " error[1]: " + str(
            self.error[1]) + " Experience: " + str(self.experience) + " numerisity: " + str(
            self.numerosity) + " weight: " + str(self.weight) + " position: " + str(self.position)

    """
       Mutates this classifier, changing the condition and action
       @param state - The state of the system to mutate around
       @param mutation_rate - The probability with which to mutate
       @param num_actions - The number of actions in the system
    """

    def _mutate(self, state, mutation_rate, num_actions):
        self.condition = ''.join(
            [self.condition[i] if numpy.random.rand() > mutation_rate else state[i] if self.condition[i] == '#' else '#'
             for i in range(len(self.condition))])
        if numpy.random.rand() < mutation_rate:
            self.action = numpy.random.randint(0, num_actions)

    """
       Calculates the deletion vote for this classifier, that is, how much it thinks it should be deleted
       @param average_fitness - The average fitness in the current action set
       @param theta_del - See parameters above
       @param delta - See parameters above
    """

    def _delete_vote(self, average_fitness, theta_del, delta):
        vote = self.average_size * self.numerosity
        if self.experience > theta_del and (
                (self.fitness[1] + self.fitness[0]) / 2) / self.numerosity < delta * average_fitness:
            #print("delete vote")
            return vote * average_fitness / (((self.fitness[1] + self.fitness[0]) / 2) / self.numerosity)
        else:
            return vote

    """
        Returns whether this classifier can subsume others
        @param theta_sub - See parameters above
        @param e0 - See parameters above
    """

    def _could_subsume(self, theta_sub, e0):
        return self.experience > theta_sub and (self.error[0] + self.error[1]) / 2 < e0

    """
        Returns whether this classifier is more general than another
        @param other - the classifier to check against
    """

    def _is_more_general(self, other):
        if len([i for i in self.condition if i == '#']) <= len([i for i in other.condition if i == '#']):
            return False

        return all([s == '#' or s == o for s, o in zip(self.condition, other.condition)])

    """
        Returns whether this classifier subsumes another
        @param other - the classifier to check against
        @param theta_sub - See parameters above
        @param e0 - See parameters above
    """

    def _does_subsume(self, other, theta_sub, e0):
        return self.action == other.action and self._could_subsume(theta_sub, e0) and self._is_more_general(other)

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        if other == None:
            return False
        return self.id == other.id


"""
   The main XCS class
"""


class moxcs:
    """
        Initializes an instance of the X classifier system
        @param parameters - A parameters instance (See above), containing the parameters for this system
        @param state_function - A function which returns the current state of the system, as a string
        @param reward_function - A function which takes a state and an action, performs the action and returns the reward
        @param eop_function - A function which returns whether the state is at the end of the problem
        @param initalize_function - A function which initailize each episode
    """

    def __init__(self, parameters, state_function, reward_function, eop_function, initalize_function, stateToCondition,env):
        self.pre_rho = 0
        self.parameters = parameters
        self.state_function = state_function
        self.reward_function = reward_function
        self.stateToCondition = stateToCondition
        self.eop_function = eop_function
        self.population = []
        self.populationbackup = []
        self.time_stamp = 0
        self.initalize_function = initalize_function
        self.previous_action_set = []
        self.temp_previous_action_set = []
        self.previous_reward = None
        self.previous_state = None
        self.initialPrediction = 5
        self.env = env
        self.env._max_episode_steps = 1000

    """
        Prints the current population to stdout
    """

    def print_population(self):
        with open('pop.csv', 'wb') as result_file:
            for i in self.population:
                result_file.write(str(i))


    """
       Classifies the given state, returning the class
       @param state - the state to classify
    """

    def classify(self, state, weight):
        match_set = self._generate_match_set(state, weight)
        predictions = self._generate_predictions_norm(match_set, weight)
        action = numpy.argmax(predictions)
        return action

    """
        Runs a single iteration/a step of the learning algorithm for this XCS instance
    """

    # def run_experiment_seperate(self):
    #     action,curr_state,predictions,action_set,reward, obs_next = self.getAction()
    #     self.updateSet(curr_state, predictions, action_set, reward)

    # def run_episode(self):
    #     self.initalize_function()
    #     while not self.eop_function():
    #         self.run_experiment_seperate()
    #todo: update state condition = self.stateToCondition(pos, vel)
    #todo: get position first!!!!!!!!!!!!!!!!!!!!!!!!!
    def generate_population(self,weights, position):
        for pos in position:
            condition = ''.join(self.state_function(self.env.getConditionXY(pos[0],pos[1],123)))
            # condition = self.stateToCondition(conditionString)
            for weight in weights:
                for action in range(self.parameters.num_actions):
                    # condition = np.array2string(np.random.randint(0,bitRange,size=bits))

                    clas = self._generate_covering_classifier_action(condition, action, weight, pos)
                    self._insert_to_population(clas)
            print(len(self.population))
        print(len(self.population))
        print(self.population)
        return self.population

    def emptyPopulation(self):
        self.population=[]

    def allHashClassifier(self,weights):
        condition = '########################'
        for weight in weights:
            clas = self._generate_covering_classifier_action(condition, 4, weight, [100,100])
            self._insert_to_population(clas)
        print(self.population)
        return self.population

    # def generate_population(self,weights):
    #     for weight in weights:
    #         for action in range(self.parameters.num_actions):
    #             for pos in range(13):
    #                 for vel in range(21):
    #                     condition = self.stateToCondition(pos, vel)
    #                     clas = self._generate_covering_classifier_action(condition, action, weight)
    #                     self._insert_to_population(clas)
    #     print(len(self.population))
    #     print(self.population)
    #     return self.population

    ##################################################################################
    def stateAction(self, weight):
        # 12 0-11 pos_bin
        # 20 numbers, 0-19 vel_bin

        actionList=[]
        predictionsList=[]
        x=[]
        y=[]

        states = []
        for pos in range(12):
            for vel in range(20):
                states.append((pos, vel))
                #print(pos, vel)
        for state in states:
            state_conditon = self.stateToCondition(state[0],state[1])
            match_set = self._generate_a_match_set_weight(state_conditon, weight)
            predictions = self._generate_predictions_norm(match_set,weight)
            action = self._select_determinstic_action(predictions)
            #print(state, action, predictions)
            x.append(state[0])
            y.append(state[1])
            actionList.append(action)
            predictionsList.append(predictions)
        #TODO: output to csv
        # 字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'x': x,'y':y, 'actionList': actionList,'predictionsList':predictionsList})

        # 将DataFrame存储为csv,index表示是否显示行名，default=True
        filename = "stateWithAction_"+str(weight)+".csv"
        dataframe.to_csv(filename, index=False, sep=',')




    # todo: weigths framework
    def run_iteration_episode(self, j, weights,ideaPoint, neighboursize):
        # get first state: curr_state
        # print("before reset")
        self.env.reset()
        # print("after reset")
        # curr_condition = self.initalize_function()
        # self.env.render()
        #todo: update first state
        # print("problem")

        print(j)


        # curr_condition = '001010010001100001100100'
        # print("curr_condition")




        # todo: 10 float to 10 integer list
        curr_condition = ''.join(self.state_function(self.env.getCondition(123)))
        curr_agentPosition = self.env.getAgentPosition(456)
        # aaa = self.env.getConditionXY(2,2,123)
        # print(aaa)
        # print("resetresetreset")
        # print(curr_agentPosition)

        # print(curr_condition)
        self.env._max_episode_steps = 1000
        step = 0
        # self.false = False
        # done = self.false
        done = False
        #ignore initailize function
        #md = moeadMethod(2, 3, (-10, -10))
        #2 obj, 11weights
        #weights = md.initailize(2, 3)
        foo1 = [0]
        temp = np.array([random.choice(foo1)])
        temp_reward = 0


        while not self.eop_function(done):
            # Hardcode weight to get match_set, as we want agent go as far as he can
            #matchset only for action selection


            match_set_action = self._generate_match_set_weight(curr_condition, weights[0], curr_agentPosition)
            predictions = self._generate_predictions_norm(match_set_action, weights[0])
            # if j>5:
            #     print(predictions)
            # get action
            # if j>90:
            #self.env.render()
            #todo:get action!!!!!!!!!
            if (random.random() < 0.5):
                action = self._select_action(predictions)
                action = np.array([action])
            else:
                foo = [0,1,2,3,4,5,6] #1：right 3:jump
                action = np.array([random.choice(foo)])
            # if step==0:
            #     action = np.array([1])
            # if step==1:
            #     action = np.array([1])
            # if step==2:
            #     action = np.array([1])
            # if step==3:
            #     action = np.array([4])
            # if step==4:
            #     action = np.array([4])
            # if step==5:
            #     action = np.array([4])
            # if step==6:
            #     action = np.array([4])
            # if step==7:
            #     action = np.array([4])
            # if step==8:
            #     action = np.array([4])
            # if step==9:
            #     action = np.array([4])
            # if step==10:
            #     action = np.array([4])
            # if step==11:
            #     action = np.array([1])
            # if step==12:
            #     action = np.array([1])
            # if step==13:
            #     action = np.array([1])
            # if step==14:
            #     action = np.array([1])
            # if step==15:
            #     action = np.array([1])
            # if step==16:
            #     action = np.array([4])


            # print("current position", self.env.getAgentPosition(123))
            # print(action)

            # if action == np.array(3):
            #     next_state, reward, done, info = self.env.step(temp)
            #     print("reward with 3")
            #     print(reward)
            #     print("done inside")
            #     print(done)

            #real matchset for generate actionset
            #match_set = self._generate_match_set(curr_condition)
            #action_set = [clas for clas in match_set if clas.action == action]
            # take aciton and get reward, go to new state
            # todo: next_state goes to mask-rcnn and return 10 float list
            count = 0
            next_state, reward, done, info = self.env.step(action)
            if temp_reward<1:
                temp_reward = reward
            # if done == True:
            #     print("DONE BY done main")
                # print("before action position", curr_agentPosition)
                # print("after action position", self.env.getAgentPosition(123))
                # print("action", action)
            # print(action)
            #*********************
            if done != True:
                done1 = False
                while count < 8:
                    if done1 != True:
                        if(count < 4):
                            next_state1, reward1, done1, info = self.env.step(action)
                            if temp_reward < 1:
                                temp_reward = reward1
                            if done1 == True:
                                # print("DONE BY count<3")
                                # print("position", self.env.getAgentPosition(123))
                                # print("action", action)
                                done = True
                                # print("count ",count)
                        else:
                            next_state1, reward1, done1, info = self.env.step(temp)
                            if temp_reward < 1:
                                temp_reward = reward1
                            if done1 == True:
                                # print("DONE BY count>3")
                                # print("position", self.env.getAgentPosition(123))
                                # print("action", action)
                                done = True
                                # print("count " ,count)
                        # print("sub action: ")
                        # print(temp)
                        count = count + 1
                    else:
                        done = True
                        count=51
                        # print("done")
            #****************************





            # print("reward outside 3")
            # print(reward)
            # print("done outside")
            #*****************************
            # if done == True:
            #     print("final done")
            #     print("before action position", curr_agentPosition)
            #     print("after action position", self.env.getAgentPosition(123))
            #     print("action", action)

            # if reward == np.array(10):
            #     done = True
            # else:
            #     done = False
            print("reward:",reward)
            print("reward1:", reward1)
            print("temp_reward:",temp_reward)

            if done == True and step<2000 and temp_reward[0]>55:
                reward= 1000
                reward= numpy.append(reward, 0)
            else:
                reward = 0
                reward= numpy.append(reward, 0)
            if done == True and step<2000 and temp_reward[0]<55:
                reward = 0
                reward = numpy.append(reward, 20)
            else:
                reward[1]=0

            # if done == True and step<2000:
            #     reward = 1000
            # else:
            #     reward = 0
            # if action==1:
            #     reward = numpy.append(reward, 10)
            # else:
            #     reward = numpy.append(reward, 0)

            # condition = CoinRunVecEnv.getCondition(123)
            #todo:step_coinrun
            condition = self.env.getCondition(123)

            # if step>1100:
            #     print(self.env.getAgentPosition(123))
                # print(self.population)
                # self.print_classifiers(self.population)
            # print("condition from getCondition")
            # print(condition)

            next_conditon = ''.join(self.state_function(condition))
            #todo:vec_wait
            next_agentPosition = self.env.getAgentPosition(456)
            # print(next_agentPosition)

            ifFirst = False
            ifLast = False

            # update set
            for i in range(len(weights)):
                match_set_w = self._generate_match_set_weight(curr_condition, weights[i], curr_agentPosition)
                # predictions_w = self._generate_predictions_norm(match_set_w, weights.last)
                predictions0 = self._generate_predictions(match_set_w, 0)
                predictions1 = self._generate_predictions(match_set_w, 1)
                predictionsnorm = self._generate_predictions_norm(match_set_w, weights[i])

                maxIndex = predictionsnorm.index(max(predictionsnorm))
                action_set_w = [clas for clas in match_set_w if (clas.weight == weights[i] and clas.action==action)]
                if weights[i] == weights[0]:
                    ifFirst = True
                #update: add ifFirst = false
                else:
                    ifFirst = False
                if weights[i] == weights[-1]:
                    ifLast = True
                else:
                    ifLast = False
                self.updateSet(next_conditon, predictions0, predictions1, maxIndex, action_set_w, reward, done, weights[i],i,
                               weights,ifFirst,ifLast,ideaPoint, neighboursize)


            curr_condition = next_conditon
            curr_agentPosition = next_agentPosition
            step=step+1
        print("step: ",step)

        # print("finish iteration ")
    def printPopulation(self):
        print(self.population)

    def savePopulation(self):
        self.populationbackup = self.population

    def importPopulation(self):
        self.population = self.populationbackup

    def set_curr_x(self,x):
        self.env.set_curr_x(x)

    def run_testing_episode(self, testWeight, j, iteration):
        # get first state: curr_state
        #print(self.population)

        foo1 = [0]
        temp = np.array([random.choice(foo1)])
        reward_final=0
        self.allHashClassifier([[1, 0]])
        self.allHashClassifier([[0.5, 0.5]])
        self.allHashClassifier([[0, 1]])

        # print("print population").
        print("testingtestingtesting")
        print(j)
        self.print_classifiers(self.population)

        self.env.reset()
        # curr_condition = self.initalize_function()
        curr_condition = '001010010001100001100100'
        # curr_condition = ''.join(self.state_function(self.env.getCondition(123)))



        self.env._max_episode_steps = 1000
        step = 0
        done = False
        stay = 0
        rewardTotal = [0,0]

        df = pd.DataFrame(columns=['actType', 'position', 'action', 'iteration','rewardget'])
        actType = []
        positionList = []
        actionList = []
        iterationList = []
        rewardget = []




        while not self.eop_function(done):
            match_set_action = self._generate_match_set_weight_without_covering(curr_condition, testWeight)
            predictions = self._generate_predictions_norm(match_set_action, testWeight)
            #get action

            # self.env.render()
            if len(match_set_action)>0:
                action = self._select_determinstic_action(predictions)
            else:
                print("cannot find matchset")
                # action = pre_action
                action = 0

            # if step==0:
            #     action = 4

            action = np.array([action])




            position = self.env.getAgentPosition(123)
            # aaa = self.env.getConditionXY(2, 2, 123)
            # print(aaa)
            print("main-action: ", position,"act: ", action)
            print("match_set_action")
            # for cl in match_set_action:
            #     print(cl)
            # print(predictions)
            actType.append('main-action')
            # print(position)
            if j < 1:
                positionList.append(position)
                print(action)
                actionList.append(action)
                iterationList.append(iteration)

            count = 0
            next_state, reward, done, info = self.env.step(action)
            rewardget.append(reward[0])
            if reward[0]>1:
                reward_final=reward[0]
            # print("main action: ")
            # print(action)
            if done != True:
                done1 = False
                while count < 8:
                    if done1 != True:
                        if (count < 4):
                            position = self.env.getAgentPosition(123)
                            next_state1, reward1, done1, info = self.env.step(action)
                            rewardget.append(reward1[0])
                            if reward1[0] > 1:
                                reward_final = reward1[0]
                            if done1 == True:
                                done = True
                            # print("sub-action-1")
                            # print(position)
                            # print(action)
                            if j < 1:
                                actType.append('sub-action-1')
                                positionList.append(position)
                                actionList.append(action)
                                iterationList.append(iteration)

                        else:
                            position = self.env.getAgentPosition(123)
                            next_state1, reward1, done1, info = self.env.step(temp)
                            rewardget.append(reward1[0])
                            if reward1[0] > 1:
                                reward_final = reward1[0]
                            # print("sub-action-2")
                            # print(position)
                            # print(temp)
                            if done1 == True:
                                done = True
                            if j < 1:
                                actType.append('sub-action-2')
                                positionList.append(position)
                                actionList.append(temp)
                                iterationList.append(iteration)
                        # print("sub action: ")
                        # print(temp)
                        count = count + 1
                    else:
                        done = True
                        count = 51
                        # print("done")
            step = step+1
            if step>2222:
                done = True



            if action==4:
                stay=stay+1
            # real matchset for generate actionset
            # match_set = self._generate_match_set(curr_condition)
            # action_set = [clas for clas in match_set if clas.action == action]
            # take aciton and get reward, go to new state
            # if done != True:
            #     next_state, reward, done, info = self.env.step(action)

            condition = self.env.getCondition(123)

            # print(position)
            # print(action)
            next_conditon = ''.join(self.state_function(condition))

            reward = numpy.append(reward, 0)
            # next_conditon = self.state_function(next_state)

            curr_condition = next_conditon

            rewardTotal = rewardTotal+reward
            pre_action = action

        if j<1:
            dataframe = pd.DataFrame(
                {'actType': actType, 'position': positionList, 'action': actionList, 'iteration': iteration, 'reward': rewardget})
            df = df.append(dataframe, ignore_index=True)

            filename = "./data/actionList" + "interation_" + str(iteration)+"_" +str(testWeight) + ".csv"
            df.to_csv(filename, index=False, sep=',')

        print("j:",j," current iteration number of steps", step)



        # with open('result.csv', 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(list)
        # dataframe = pd.DataFrame({'validation problem':j,'step': step, 'testWeight': testWeight})

        return step,stay,reward_final

    # def getAction(self):
    #     # get current state
    #     curr_state = self.state_function()
    #     match_set = self._generate_match_set(curr_state)
    #     predictions = self._generate_predictions(match_set)
    #
    #     action = self._select_action(predictions)
    #     action_set = [c
    #     las for clas in match_set if clas.action == action]
    #     reward, obs_next = self.reward_function(curr_state, action)
    #     return action,curr_state,predictions,action_set,reward, obs_next

    def updateSet(self, next_state, predictions0, predictions1, maxIndex, action_set, reward, done, weight, i, weights, ifFirst, ifLast,ideaPoint, neighboursize):
        # reward=[0,0]  :  r[0]是每走一步都需要减一，r[1] 减一for踩刹车和加速 （类似拿到大的奖励）
        indexList=[]
        moeadWeights=[]
        neighbourIndex = weights.index(weight)
        for i in range(neighboursize):
            if i%2==0:
                neighbourIndex = neighbourIndex+i
            if i%2==1:
                neighbourIndex = neighbourIndex-i
            indexList.append(neighbourIndex)

        for index in indexList:
            if index>=0 and index<=len(indexList)-1:
                moeadWeights.append(weights[index])

        P = [0, 0]
        updateSet=[]
        # if previous actionset is not null, update previous action set
        if len(self.previous_action_set) > 0:
            #updateSet = [clas for clas in self.previous_action_set if clas.weight == weight]
            for mweight in moeadWeights:
                updateSet.extend([clas for clas in self.previous_action_set if clas.weight == mweight])

        if updateSet:
            # P[0]:step
            # Qplus1 = 1 + self.parameters.gamma * max(predictions0)
            if reward[0]>0:
                P[0] = reward[0]
                P[1] = reward[1]
            else:
                P[0] = reward[0] + self.parameters.gamma * predictions0[maxIndex]
            # P[1]:reward
                P[1] = reward[1] + self.parameters.gamma * predictions1[maxIndex]

            # P = self.previous_reward + self.parameters.gamma * max(predictions_w)

            self._update_set(updateSet, P)
            #self._run_ga(updateSet, self.previous_state, ideaPoint)

        if self.eop_function(done):
            # if reward[0] != 0:
            #     print("eop_function, reward[0] is ", reward[0])
            #TODO:CHANGE P[0] = 1
            P[0] = reward[0]

            P[1] = reward[1]

            self._update_set(action_set, P)
            # self._run_ga(action_set, curr_state)
            if i==len(weights)-1:
                self.previous_action_set = []
                self.temp_previous_action_set=[]
        else:
            if ifFirst:
                self.temp_previous_action_set=action_set
            else:
                self.temp_previous_action_set.extend(action_set)
            if ifLast:
                self.previous_action_set=self.temp_previous_action_set
                self.temp_previous_action_set =[]
            self.previous_reward = reward
            self.previous_state = next_state

        self.time_stamp = self.time_stamp + 1

    """
        Generates the match set for the given state, covering as necessary
        @param state - the state to generate a match set for
    """
    #todo: state is a list with 10 float
    def _generate_match_set_weight(self, state, weight=[], curr_agentPosition=[]):
        set_m = []
        while len(set_m) == 0:
            set_m = [clas for clas in self.population if state_matches(clas.condition, state) and clas.weight == weight]
            if len(set_m) < self.parameters.theta_mna:  # Cover
                clas = self._generate_covering_classifier(state, set_m, weight, curr_agentPosition)
                self._insert_to_population(clas)
                self._delete_from_population()
                set_m = []
        return set_m

    def _generate_match_set_weight_without_covering(self, state, weight=[], curr_agentPosition=[]):
        return [clas for clas in self.population if state_matches(clas.condition, state) and clas.weight == weight]

    def _generate_match_set(self, state):
        set_m = []
        while len(set_m) == 0:
            set_m = [clas for clas in self.population if state_matches(clas.condition, state)]
            if len(set_m) < self.parameters.theta_mna:  # Cover
                clas = self._generate_covering_classifier(state, set_m)
                self._insert_to_population(clas)
                self._delete_from_population()
                set_m = []
        return set_m
    """
        Deletes a classifier from the population, if necessary
    """

    # TODO:multi, depends on matchset filter
    # comment remove from population so far
    def _delete_from_population(self):
        numerosity_sum = sum([clas.numerosity for clas in self.population])
        # 如果numerosity_sum没有达到门限值 不删
        if numerosity_sum <= self.parameters.N:
            return

        average_fitness = sum([clas.fitnessAvg for clas in self.population]) / numerosity_sum
        votes = [clas._delete_vote(average_fitness, self.parameters.theta_del, self.parameters.delta) for clas in
                 self.population]
        vote_sum = sum(votes)
        choice = numpy.random.choice(self.population, p=[vote / vote_sum for vote in votes])
        if choice.numerosity > 1:
            choice.numerosity = choice.numerosity - 1
        else:
             self.population.remove(choice)

    """
        Inserts the given classifier into the population, if it isn't able to be
        subsumed by some other classifier in the population
        @param clas - the classifier to insert
    """

    def _insert_to_population(self, clas):
        if len(self.population)==0:
            same = 0

        else:
            same = [c for c in self.population if
                    # (c.action, c.condition, c.weight) == (clas.action, clas.condition, clas.weight)
                     (c.action==clas.action) and (c.condition==clas.condition) and (c.weight==clas.weight)]
        if same:
            same[0].numerosity = same[0].numerosity + 1
        else:
            self.population.append(clas)

        return self.population

    """
        Generates a classifier that conforms to the given state, and has an unused action from
        the given match set
        @param state - The state to make the classifier conform to
        @param match_set - The set of current matches
    """

    def _generate_covering_classifier(self, state, match_set, weight=[], position =[0,0]):
        clas = classifier(self.parameters, state, weight)
        used_actions = [classifier.action for classifier in match_set]
        available_actions = list(set(range(self.parameters.num_actions)) - set(used_actions))
        clas.action = numpy.random.choice(available_actions)
        clas.time_stamp = self.time_stamp
        clas.position = position
        return clas

    def _generate_covering_classifier_action(self, state, action, weight=[], position =[0,0]):
        clas = classifier(self.parameters, state, position)
        clas.action = action
        clas.weight = weight
        clas.time_stamp = self.time_stamp
        clas.position = position
        return clas

    """
        Generates a prediction array for the given match set
        @param match_set - The match set to generate predictions for
    """

    def _generate_predictions(self, match_set, obj):
        PA = [0] * self.parameters.num_actions
        FSA = [0] * self.parameters.num_actions

        for clas in match_set:
            if obj == 0:
                try:
                    PA[clas.action] += clas.prediction[0] * clas.fitness[0]
                    FSA[clas.action] += clas.fitness[0]
                except:
                    print("aaa")
                    PA[clas.action] += clas.prediction[0] * clas.fitness[0]
                    FSA[clas.action] += clas.fitness[0]
            if obj == 1:
                PA[clas.action] += clas.prediction[1] * clas.fitness[1]
                FSA[clas.action] += clas.fitness[1]

        normal = [PA[i] if FSA[i] == 0 else PA[i] / FSA[i] for i in range(self.parameters.num_actions)]
        #print(normal)

        return normal

    def _generate_predictions_withoutF(self, match_set, obj):
        PA = [0] * self.parameters.num_actions
        # FSA = [0] * self.parameters.num_actions

        for clas in match_set:
            if obj == 0:
                try:
                    PA[clas.action] = clas.prediction[0]
                    # FSA[clas.action] += clas.fitness[0]
                except:
                    print("aaa")
                    PA[clas.action] = clas.prediction[0]
                    # FSA[clas.action] += clas.fitness[0]
            if obj == 1:
                PA[clas.action] = clas.prediction[1]
                # FSA[clas.action] += clas.fitness[1]

        normal = [PA[i] for i in range(self.parameters.num_actions)]
        #print(normal)

        return normal

    ########################################################################################################

    def _generate_predictions_norm(self, match_set, weight):
        predictions0 = self._generate_predictions(match_set, 0)
        predictions1 = self._generate_predictions(match_set, 1)

        # normalisation
        #todo: update stepNor
        #predictions0_norm = [self.stepNor(q, 1000) for q in predictions0]
        predictions0_norm = predictions0
        #predictions1_norm = [self.stepNor(q, 1000) for q in predictions1]
        predictions1_norm = predictions1

        #total
        predictionTotal = self.getTotalPrediciton(weight, predictions0_norm, predictions1_norm)

        return predictionTotal



    def _generate_predictions_norm_withoutF(self, match_set, weight):
        predictions0 = self._generate_predictions_withoutF(match_set, 0)
        predictions1 = self._generate_predictions_withoutF(match_set, 1)

        # normalisation
        #todo: update stepNor
        #predictions0_norm = [self.stepNor(q, 1000) for q in predictions0]
        predictions0_norm = predictions0
        #predictions1_norm = [self.stepNor(q, 1000) for q in predictions1]
        predictions1_norm = predictions1

        #total
        predictionTotal = self.getTotalPrediciton(weight, predictions0_norm, predictions1_norm)

        return predictionTotal

    def stepNor(self, q, max):
        #return abs(max + 1 - q) / max
        return abs(max + q) / max

    def rewardNor(self, q, max, min):
        return (q - min) / (max - min)

    def getTotalPrediciton(self, weight, predictions0, predictions1):
        totalPre = [1] * predictions0.__len__()
        for i in range(predictions0.__len__()):
            totalPre[i] = weight[0] * predictions0[i] + weight[1] * predictions1[i]
        return totalPre

    """
        Selects the action to run from the given prediction array. Takes into account exploration
        vs exploitation
        @param predictions - The prediction array to generate an action from
    """

    def _select_action(self, predictions):
        valid_actions = [action for action in range(self.parameters.num_actions) if predictions[action] != 0]
        if len(valid_actions) == 0:
            return numpy.random.randint(0, self.parameters.num_actions)

        if numpy.random.rand() < self.parameters.prob_exploration:
            return numpy.random.choice(valid_actions)
        else:
            return numpy.argmax(predictions)

    """
           Selects the action to run from the given prediction array. Takes into account exploration
           vs exploitation
           @param predictions - The prediction array to generate an action from
       """

    def _select_determinstic_action(self, predictions):
        valid_actions = [action for action in range(self.parameters.num_actions) if predictions[action] != 0]
        if len(valid_actions) == 0:
            return numpy.random.randint(0, self.parameters.num_actions)
        else:
            return numpy.argmax(predictions)

    """
       Updates the given action set's prediction, error, average size and fitness using the given decayed performance
       @param action_set - The set to update
       @param P - The reward to use
    """

    def _update_set(self, action_set, P):
        set_numerosity = sum([clas.numerosity for clas in action_set])
        for clas in action_set:
            clas.experience = clas.experience + 1
            for i in range(2):
                if clas.experience < 1. / self.parameters.beta:
                    clas.prediction[i] = clas.prediction[i] + (P[i] - clas.prediction[i]) / clas.experience
                    clas.error[i] = clas.error[i] + (abs(P[i] - clas.prediction[i]) - clas.error[i]) / clas.experience
                    if i == 0:
                        clas.average_size = clas.average_size + (set_numerosity - clas.numerosity) / clas.experience
                else:
                    clas.prediction[i] = clas.prediction[i] + (P[i] - clas.prediction[i]) * self.parameters.beta
                    clas.error[i] = clas.error[i] + (
                                abs(P[i] - clas.prediction[i]) - clas.error[i]) * self.parameters.beta
                    if i == 0:
                        clas.average_size = clas.average_size + (
                                    set_numerosity - clas.numerosity) * self.parameters.beta

        # Update fitness

        kappa0 = {clas: 1 if clas.error[0] < self.parameters.e0 else self.parameters.alpha * (
                clas.error[0] / self.parameters.e0) ** -self.parameters.nu for clas in action_set}
        kappa1 = {clas: 1 if clas.error[1] < self.parameters.e0 else self.parameters.alpha * (
                clas.error[1] / self.parameters.e0) ** -self.parameters.nu for clas in action_set}
        accuracy_sum0 = sum([kappa0[clas] * clas.numerosity for clas in action_set])
        accuracy_sum1 = sum([kappa1[clas] * clas.numerosity for clas in action_set])

        for clas in action_set:
            clas.fitness[0] = clas.fitness[0] + self.parameters.beta * (
                    kappa0[clas] * clas.numerosity / accuracy_sum0 - clas.fitness[0])
            clas.fitness[1] = clas.fitness[1] + self.parameters.beta * (
                    kappa1[clas] * clas.numerosity / accuracy_sum1 - clas.fitness[1])

            clas.fitnessAvg = (clas.fitness[0] + clas.fitness[1]) / 2

        #print(action_set)

        if self.parameters.do_action_set_subsumption:
            self._action_set_subsumption(action_set);

    """
        Does subsumption inside the action set, finding the most general classifier
        and merging things into it
        @param action_set - the set to perform subsumption on
    """

    def _action_set_subsumption(self, action_set):
        cl = None
        for clas in action_set:
            if clas._could_subsume(self.parameters.theta_sub, self.parameters.e0):
                if cl == None or len([i for i in clas.condition if i == '#']) > len(
                        [i for i in cl.condition if i == '#']) or numpy.random.rand() > 0.5:
                    cl = clas

        if cl:
            for clas in action_set:
                if cl._is_more_general(clas):
                    cl.numerosity = cl.numerosity + clas.numerosity
                    try:
                        # print("Print action_set before deletion")
                        # self.print_classifiers(action_set)
                        # print("Print population before deletion")
                        # self.print_classifiers(self.population)
                        # print("delete class:")
                        # print(clas)
                        if clas.numerosity > 1:
                            clas.numerosity = clas.numerosity - 1
                            # print("delete clas numerosity in action_set")
                        else:
                            action_set.remove(clas)
                            # print("delete clas in action_set")
                        # print("finish delete clas in action_set")
                        # self.print_classifiers(action_set)


                    # If delete from action set not need to remove it from population
                    # if clas.numerosity>1:
                    #     clas.numerosity=clas.numerosity-1
                    #     print("delete clas numerosity in population")
                    # else:
                    #     self.population.remove(clas)
                    #     print("delete clas in population")
                    # print("finish delete clas in population")
                    # self.print_classifiers(self.population)
                    except:
                        print("An exception occurred")

    def print_classifiers(self, classifiers):
        list=[]
        for clas in classifiers:
            list.append([clas])
        with open('./data/data.csv', 'w+') as f:
            writer = csv.writer(f, delimiter ='\t')
            writer.writerows(list)



    """
        Runs the genetic algorithm on the given set, generating two new classifers
        to be inserted into the population
        @param action_set - the action set to choose parents from
        @param state - The state mutate with
    """


    def _run_ga(self, action_set, state, ideaPoint):
        if len(action_set) == 0:
            return

        if self.time_stamp - numpy.average([clas.time_stamp for clas in action_set],
                                           weights=[clas.numerosity for clas in action_set]) > self.parameters.theta_ga:
            for clas in action_set:
                clas.time_stamp = self.time_stamp

            fitness_sum = sum([clas.fitnessAvg for clas in action_set])

            probs = [clas.fitnessAvg / fitness_sum for clas in action_set]
            parent_1 = numpy.random.choice(action_set, p=probs)
            parent_2 = numpy.random.choice(action_set, p=probs)
            child_1 = deepcopy(parent_1)
            child_2 = deepcopy(parent_2)
            child_1.id = classifier.global_id
            child_2.id = classifier.global_id + 1
            classifier.global_id = classifier.global_id + 2
            child_1.numerosity = 1
            child_2.numerosity = 1
            child_1.experience = 0
            child_2.experience = 0

            if numpy.random.rand() < self.parameters.crossover_rate:
                _crossover(child_1, child_2)
                for i in range(2):
                    child_1.prediction[i] = child_2.prediction[i] = numpy.average([parent_1.prediction[i], parent_2.prediction[i]])
                    child_1.error[i] = child_2.error[i] = numpy.average([parent_1.error[i], parent_2.error[i]])
                    child_1.fitness[i] = child_2.fitness[i] = numpy.average([parent_1.fitness[i], parent_2.fitness[i]])

            child_1.fitnessAvg = (child_1.fitness[0]+child_1.fitness[1])*0.1/2
            child_2.fitnessAvg = (child_2.fitness[0] + child_2.fitness[1])*0.1/2

            for child in [child_1, child_2]:
                child._mutate(state, self.parameters.mutation_rate, self.parameters.num_actions)

                #TODO: use ideal point to evaluate child
                #update ideaPoint
                for i in range(2):
                    child.prediction[i] > ideaPoint[i]
                    ideaPoint[i] = child.prediction[i];


                if self.parameters.do_GA_subsumption == True:
                    if parent_1._does_subsume(child, self.parameters.theta_sub, self.parameters.e0):
                        parent_1.numerosity = parent_1.numerosity + 1
                    elif parent_2._does_subsume(child, self.parameters.theta_sub, self.parameters.e0):
                        parent_2.numerosity = parent_2.numerosity + 1
                    else:
                        self._insert_to_population(child)
                else:
                    self._insert_to_population(child)

                self._delete_from_population()


"""
    Returns whether the given state matches the given condition
    @param condition - The condition to match against
    @param state - The state to match against
"""


def state_matches(condition, state):
    # 返回all true，c, s从condition,state里面各取一个字符
    return all([c == '#' or c == s for c, s in zip(condition, state)])
    #todo:condition is float[10], state is int[10], each int is between 1 to 16 (0-64 16个区间,0 means not match)
    # return condition==state



"""
    Cross's over the given children, modifying their conditions
    @param child_1 - The first child to crossover
    @param child_2 - The second child to crossover
"""


def _crossover(child_1, child_2):
    x = numpy.random.randint(0, len(child_1.condition))
    y = numpy.random.randint(0, len(child_1.condition))

    child_1_condition = list(child_1.condition)
    child_2_condition = list(child_2.condition)

    for i in range(x, y):
        child_1_condition[i], child_2_condition[i] = child_2_condition[i], child_1_condition[i]

    child_1.condition = ''.join(child_1_condition)
    child_2.condition = ''.join(child_2_condition)
