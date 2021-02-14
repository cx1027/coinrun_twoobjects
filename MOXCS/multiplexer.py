import numpy.random
import xcsbackup

"""
    An implementation of an N-bit multiplexer problem for the X classifier system
"""

#The number of bits to use for the address in the multiplexer, 3 bit in example
#bits=3, then state_length=12 100%
#bits=4, length=20
bits = 4

#The maximum reward
rho = 1000

#The number of steps we learn for
learning_steps = 20000

#The number of steps we validate for
validation_steps = 20000

"""
    Returns a random state of the multiplexer
"""
def state():
    return ''.join(['0' if numpy.random.rand() > 0.5 else '1' for i in range(0, bits + 2**bits)])

"""
    The N-bit multiplexer is a single step problem, and thus always is at the end of the problem
"""
def eop():
    return True

"""
    Calculates the reward for performing the action in the given state
"""
def reward(state, action):
    #Extract the parts from the state
    address = state[:bits]
    data = state[bits:]

    #Check the action
    if str(action) == data[int(address, 2)]:
        return rho
    else:
        return 0

#Set some parameters
parameters = xcsbackup.parameters()
parameters.state_length = bits + 2**bits
parameters.theta_mna = 2
parameters.e0 = 1000 * 0.01
parameters.theta_ga = 25
parameters.gamma = 0
parameters.N = 400000

#Construct an XCS instance
my_xcs = xcsbackup.xcs(parameters, state, reward, eop)

#Train
for j in range(learning_steps):
    #for each step
    #TODO:seperate get action and update Set
    my_xcs.run_experiment()
    #my_xcs.run_experiment_seperate()

#Validate
this_correct = 0
for j in range(validation_steps):
    rand_state = state()
    this_correct = this_correct + reward(rand_state, my_xcs.classify(rand_state))

print("Performance " + ": " + str((this_correct / validation_steps / rho) * 100) + "%");
