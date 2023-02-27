# https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1

from numpy import exp, array, random, dot

# numpy.dot(a, b, out=None) - Dot product of two arrays.
# random.random() - Return the next random floating point number in the range 0.0 <= X < 1.0
# random.seed() - By default the random number generator uses the current system time, or from a specific seed value

training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
test_example = array([1, 0, 0])

iterations = 1000

random.seed(1) # random: random.seed()
synaptic_weights = 2 * random.random((3, 1)) - 1
print(synaptic_weights)
for iteration in range(iterations):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    print(output)
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
print(synaptic_weights)

print( 1 / (1 + exp(-(dot(test_example, synaptic_weights)))))
