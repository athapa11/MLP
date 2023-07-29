import pandas
import numpy
import random
import matplotlib.pyplot as plt

# read standardised data from excel file:
training_data = pandas.read_excel('C:\Training-Data.xlsx')
validation_data = pandas.read_excel('C:\Validation-Data.xlsx')
testing_data = pandas.read_excel('C:\Testing-Data.xlsx')

# round data to 4 dec places:
training_data = training_data.round(decimals=4)
validation_data = validation_data.round(decimals=4)
testing_data = testing_data.round(decimals=4)

# convert data to numpy array:
training_data = training_data.to_numpy()
validation_data = validation_data.to_numpy()
testing_data = testing_data.to_numpy()

sample = 0 # data point counter

# parameters initialisation:
lr = float(input('Enter learning parameter: ')) # learning rate w/o annealing
epochs = int(input('Enter number of epochs: '))
num_inputs = int(input('Enter number of inputs per node: '))
hidden_nodes = int(input('Enter number of hidden nodes: '))

# weights and biases initialisation:
lower_bound = -2/num_inputs
upper_bound = 2/num_inputs

# weights for inputs per hidden node
inp_weights = numpy.array([[random.uniform(lower_bound, upper_bound) for i in range(num_inputs)] for j in range(hidden_nodes)])

# weights for hidden nodes to output node
out_weights = numpy.array([random.uniform(lower_bound, upper_bound) for i in range(num_inputs)])

# biases for hidden nodes
hidden_biases = numpy.array([random.uniform(lower_bound, upper_bound) for i in range(num_inputs)])

output_bias = random.uniform(lower_bound, upper_bound)

# total weights and biases for Omega in the weight decay:
total_weights_biases = inp_weights.size + out_weights.size + hidden_biases.size + 1


# Activation Function (Sigmoid)
def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

# Sigmoid Derivative
def sigmoid_derivative(x):
    return x*(1-x)

# Activation Function (Tanh)
def tanh(x):
    t = (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))
    return t

def tanh_derivative(t):
    dt = 1 - t**2
    return dt

# Annealing
def annealing(cycle, epochs):
    start_parm = 0.1 # start learning parameter
    end_parm = 0.01 # end learning parameter
    lr = end_parm + (start_parm - end_parm)*(1 - (1 / (1 + numpy.exp(10-((20*(cycle+1))/epochs)))))
    return lr


# Forward pass function
def forward_pass(sample, dataset):
    activated_hidden_layer = numpy.array([]) # activated hidden layer node values
    for node in range(hidden_nodes):
        # sum of hidden node - bias + sum of input*weight
        sum = hidden_biases[node]

        for inp in range(num_inputs):
            sum += dataset[sample][inp] * inp_weights[node][inp]
        activated_sum = sigmoid(sum) # activate sum per node sigmoid/tanh 

        # append value of activated hidden layer nodes:
        activated_hidden_layer = numpy.append(activated_hidden_layer, activated_sum)

    # sum of output = bias + sum of activated hidden layer nodes * hidden layer nodes weight
    output = numpy.sum(activated_hidden_layer * out_weights) + output_bias
    activated_output = sigmoid(output) # activate the output sigmoid/tanh

    return activated_output, activated_hidden_layer


# Backward pass function
def backward_pass(activated_output, activated_hidden_layer, sample, lr, cycle):
    hidden_deltas = numpy.array([]) # Delta values of hidden layer nodes
    act_node = 1 # node counter
    activation_derivative_output = sigmoid_derivative(activated_output) # or tanh derivative

    # Delta value for output = (correct - activated output) * activation function derivative of activated outputs
    delta_output = (training_data[sample][-1] - activated_output) * activation_derivative_output
    # + (1/(lr*cycle+1))*0.5*total_weights_biases**2) to add weight decay

    while act_node <= hidden_nodes:
        # Delta value for hidden cell = weight * output delta * hidden cell activation function derivative
        delta = out_weights[act_node-1]*delta_output*sigmoid_derivative(activated_hidden_layer[act_node-1]) # or tanh derivative
        hidden_deltas = numpy.append(hidden_deltas, delta)
        act_node += 1
    return delta_output, hidden_deltas


# adjust weights
def update_weights(hidden_deltas, activated_hidden_layer, ino_weights, delta_output, sample, lr):
    # save last weights to find difference for momentum
    old_inp_weights = inp_weights.copy()
    old_out_weights = out_weights.copy()

    for node in range(hidden_nodes):
        for inp in range(num_inputs):
            # new weight = old weight + lr * hidden nodes delta value * input + momentum rate * weight change
            inp_weights[node][inp] += lr * hidden_deltas[node] * training_data[sample][inp] 
            # + 0.9(inp_weights[node] - old_out_weights[node]) for weight decay

        # new output = old output + lr * output delta value * activated hidden layer nodes + momentum rate * weight change
        out_weights[node] += lr * delta_output * activated_hidden_layer[node] 
        # + 0.9(out_weights[node] - old_out_weights[node]) for weight decay


# adjust biases
def update_biases(hidden_deltas, delta_output, lr):
    global output_bias
    for node in range(hidden_nodes):
        # new bias = old bias + lr * hidden cell delta value * 1
        hidden_biases[node] += lr * hidden_deltas[node] * 1

    # new bias = old bias + lr * hidden cell delta value * 1
    output_bias += lr * delta_output * 1
    return output_bias


if __name__ == "__main__":
    # store errors for plotting graphs:
    training = numpy.array([])
    validation = numpy.array([])
    testing = numpy.array([])

    for cycle in range(epochs):
        # lr = annealing(cycle, epochs) # Annealing
        # for RMSE calculation:
        training_diff = 0
        validation_diff = 0
        sample = 0 # training sample count

        for sample in range(len(training_data)): # backpropagation through training data
            # forward pass
            activated_output, activated_hidden_layer = forward_pass(sample, training_data)

            # backward pass
            delta_output, hidden_deltas = backward_pass(activated_output, activated_hidden_layer, sample, lr, cycle)

            # update weights and biases
            update_weights(hidden_deltas, activated_hidden_layer, inp_weights, delta_output, sample, lr)
            output_bias = update_biases(hidden_deltas, delta_output, lr)

            # sum of (modelled - observed) squared for RMSE calculation
            training_diff += (activated_output - training_data[sample][-1])**2

        # RMSE calculation and saving:
        training_error = numpy.sqrt(training_diff/len(training_data)) 
        training = numpy.append(training, training_error)


        for sample in range(len(validation_data)): # predict with validation data
            # forward pass
            activated_output, activated_hidden_layer = forward_pass(sample, validation_data)

            # RMSE calculation
            validation_diff += (activated_output - validation_data[sample][-1])**2

        # RMSE calculation and saving:
        validation_error = numpy.sqrt(validation_diff/len(validation_data)) 
        validation = numpy.append(validation, validation_error)


        # RMSE at each epoch interval
        if cycle % 10 == 0:
            print(f"Epoch: {cycle}\t Training Error: {training_error}\t Validation Error: {validation_error}")

    # Training / Validation error graphs
    plt.plot(training)
    plt.plot(validation)
    plt.legend(["Training", "Validation"])
    plt.ylabel('Error', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.show()

    for sample in range(len(testing_data)): # predict for testing data
        # forward pass
        activated_output, activated_hidden_layer = forward_pass(sample, testing_data)
        if cycle == 200: # show output predictions for final model
            print(activated_output)