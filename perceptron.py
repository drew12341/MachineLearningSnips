import random
from random import choice
from numpy import array, dot, random

'''

'''

unit_step = lambda x: 0 if x < 0 else 1
# Create some training data
training_data_or = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 1),
    (array([1, 0, 1]), 1),
    (array([1, 1, 1]), 1),
]

training_data_and = [
    (array([0, 0, 1]), 0),
    (array([0, 1, 1]), 0),
    (array([1, 0, 1]), 0),
    (array([1, 1, 1]), 1),
]

training_data_not = [
    (array([0]), 1),
    (array([1]), 0),
]


def run_perceptron(training_data):
    errors = []
    bias = 0.2
    steps = 100
    # Make the randomized vector length dynamic
    vector_length = len(training_data[0][0])
    # Get a random 3-vector between 0 and 1
    # e.g. [0.03249, 0.12452, 0.49032]
    # This is used as the starting point
    rand_vec3 = random.rand(vector_length)
    print('\nStarting seed vector: {}'.format(rand_vec3))

    for _ in xrange(steps):
        vec3, expected = choice(training_data)
        print vec3, expected
        # Get the dot product of the randomized vector and the training vector
        result = dot(rand_vec3, vec3)
        # Get the offset of the expected and the unit step value
        offset = expected - unit_step(result)
        errors.append(offset)
        # Update the starting vector
        print "types", type(rand_vec3), type(bias), type(offset),type(vec3)
        rand_vec3 += bias * offset * vec3

    # Run it for visualization of the progress
    for vec3, expected in training_data:
        result = dot(vec3, rand_vec3)
        print("{}: {} = {} (expected {})".format(
            vec3[:2], result, unit_step(result), expected))


# Depending on the number of `steps` set, this may not return the right
# answer each time. Since the starting vector is random, if the step
# count is too low, it may be entirely wrong. You can play around with
# this and see where things go.
#run_perceptron(training_data_or)
#run_perceptron(training_data_and)
# This one trains much faster, as the number of cases is halved.
#run_perceptron(training_data_not)


values = [[1,2],[2,3], [2,4]]
train = [2,3,2]
weights = array([2,3])

eta = 0.1
unit_step = lambda x: 0 if x < 1 else 1
for v, r in zip(values, train):

    vec = array(v)
    #print "values", type(vec), type(r), type(weights)
    result = dot(weights, vec)
    #print "result", result,"unit step", unit_step(result)
    error = r - unit_step(result)
    #print "error", error
    #print "weights", weights
    #print "TYPES", type(weights), type(eta), type(error),type(vec)
    weights += eta * error * vec

input = array([1,1])

h1 = array([0.6,0.6])
h2 = array([1.1,1.1])



d1 = array([unit_step(dot(h1, input)), unit_step(dot(h2, input))])
print "d1", d1

output = array([-2,1.1])

print array([unit_step(dot(d1, output))])



