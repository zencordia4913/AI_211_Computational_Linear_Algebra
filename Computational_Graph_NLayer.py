import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases randomly
        self.X_for = None
        self.A_for = np.random.randn(input_size, output_size)
        self.b_for = np.random.randn(output_size)
        self.mult_for = None
        self.plus_for = None
        self.fx_for = None
        self.a = None
        self.c = None
        self.d = None
        self.e = None
        self.g = None

    def ReLu(self, x, direction, leaky_param = 0.01):
        # ReLu activation function
        if direction == "forward":
            return x * np.where(x < 0, leaky_param, 1)
        
        elif direction == "backward":
            grad = np.ones_like(x)  # Initialize gradient as ones
            grad[x < 0] = 0          # Set gradient to 0 where x < 0
            return grad

    def forward_prop(self, x):
        self.X_for = x
        self.mult_for = np.dot(self.A_for,x)
        self.plus_for = self.b_for + self.mult_for
        self.fx_for = self.ReLu(self.plus_for, "forward")
        return self.fx_for
    
    def backward_prop(self, grad, layer, max_layer):
        if layer == max_layer:
            shp = self.fx_for.shape
            grad_Z = np.ones(shp)
        else:
            print("not max")
            grad_Z = grad

        self.g = np.dot(grad_Z,self.ReLu(self.fx_for, "backward"))
        self.e = self.g
        self.d = self.g
        self.a = np.dot(self.e,(self.X_for).T)
        self.c = np.dot((self.d).T,self.A_for)
        return self.c

class Neural_Network:
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_pass(self, x):
        i = 1
        # Perform a forward pass for all the layers
        for layer in self.layers:
            x = layer.forward_prop(x)
            print(f"Output for layer {i}: ", x)
            i += 1
        return x
    
    def backward_pass(self, grad = None):
        
        i = len(self.layers) + 1
        #Perform a backward pass for all the layers
        for layer in self.layers:
            grad = layer.backward_prop(grad, i, len(self.layers) + 1)
            print(f"Gradient for layer {i}: ", grad)
            i -= 1
        return grad
        
def Computational_Graph_NN(x_input, x_size, layer_num):
    network = Neural_Network()
    for i in range(layer_num):
        if i == 0:
            output_size = int(input(f"Enter output size for layer {i + 1}: "))
            network.add_layer(Layer(input_size=x_size, output_size=output_size))
        else:
            input_size = output_size
            output_size = int(input(f"Enter output size for layer {i + 1}: "))
            network.add_layer(Layer(input_size=input_size, output_size=output_size))

    # Perform forward and backward passes
    output = network.forward_pass(x_input)
    gradient = network.backward_pass()
    return output, gradient, network

x_size = int(input("How many rows do you want in your input vector: "))
layer_num = int(input("How many layers do you want in your neural network: "))
x_input = np.random.rand(x_size)
output, gradient, network = Computational_Graph_NN(x_input, x_size, layer_num)
for layer in network.layers:
    print(vars(layer))
print("output after ", layer_num, " layers: ", output)
print("gradient after ", layer_num, " layers: ", gradient)



    
        
        
    
        
    