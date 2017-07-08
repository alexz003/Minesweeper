import math
import numpy as np

import Minesweeper

class NeuralNetwork():
	
	def __init__(self, game):
		self.weights = np.zeros((0, 0), dtype=np.float)
		self.game = game
		self.num_layers = 1
		self.num_trials = 10000
		
	# Sigmoid with vectors
	def sigmoid(self, x):
		return 1/(1 + math.exp(x))

	# Sigmoid derivative with vectors
	def div_sigmoid(x):
		return sigmoid(x)*(1-sigmoid(x))

	# Create the neural net model
	def create_model(self, num_inputs, nodes_per_layer, num_layers=1, num_trials=10000):
		# NxM weights matrix with random initial values
		self.weights = np.random.uniform(-1, 1, size=(num_layers, num_inputs*nodes_per_layer))
	
	# Runs all trials and trains the network	
	def run_all_trials(self, game, num_trials=10000):

		for i in range(num_trials):
			# Make new game if not created or not running
			if game.getBoard() == None or not game.gameStatus() == 0:
				game = Minesweeper.Minesweeper()
				game.start_game()	
	
			# Train this test case	
			self.weights = self.train(game, self.weights)	 
		
	# Feedforward on weights with board and return outputs
	def test(self, game, weights):
	
		# Get current board state	
		board = game.getBoard().flatten()
	
		# TODO(Alex): Currently only outputs total out. To save future computation,
		#		we can save the net input and output of each layer and output
		#		those arrays instead
		
		# Weights are NxM while board is 1xM
		num_layers = weights.shape[0]
	
		# Create net inputs and outputs containing the results of each layer
		net_inputs = np.zeros((num_layers, game.N*game.N), dtype=np.float)
		outputs = np.zeros(net_inputs.shape, dtype=np.float)
	
	
		# Go through each layer
		for i in range(num_layers):

			# Get's the weights in each layer
			temp_weights = weights[i,:]

			# First layer run needs to use board as input
			if i == 0:
				layer_1 = board				
					
			# Following layer runs need to use the output of the previous layer as input
			else:
				layer_1 = outputs[i-1,:]

			# Add each input and output its list
			for j in range(outputs.shape[1]):
				for k in range(outputs.shape[1]):
					# Calculate weight index: index = output_index*num_outputs + input_index
					index = j*outputs.shape[1] + k

					# Add each layer_1*weight[i, index] value to the input list
					net_inputs[i, j] += weights[i, index]*layer_1[k]

				# Add each sigmoid(layer_1) value to the output list
				outputs[i, j] = self.sigmoid(net_inputs[i,j])

		# Return the outputs
		return outputs, net_inputs

	def train(self, game, weights):

		# Get the new game board state
		board = game.getBoard().flatten()

		# Feed forward for output values
		outputs, net_inputs = self.test(game, weights)

	#	print outputs

		# Get the best output index
		index = np.random.randint(outputs.shape[1])
		for i in range(outputs.shape[1]):
			if outputs[outputs.shape[0]-1, i] > outputs[outputs.shape[0]-1, index]:
				index = i;

		if np.random.random() < .1:
			print 'random location chosen'
			index = np.random.randint(outputs.shape[1])

		# Gets location of move based on inputs
		x = int(math.floor(index/game.N))
		y = index % game.N
		
		# Sends move to the game and retrieves new board state
		game.send_move(x, y)
		new_board = game.getBoard().flatten()

		# Set up vector of expected values
		expected_values = np.zeros(outputs.shape[1], dtype = np.float)

		# Supervised Learning - expected values of weights
		# Good move:	1 - Board changes and not a bomb
		# Bad move:	0 - Bomb found
		for i in range(len(board)):
			if not board[i] == new_board[i]:
				expected_values[i] = 1

#		print new_board
#		print expected_values
		# Get new weights after backpropagating
		new_weights = self.backpropagate(board, weights, outputs, expected_values)

		# Calculate weight differences
		old_error = self.calc_total_error(expected_values, outputs)
		new_outputs, _ = self.test(game, new_weights)
		new_error = self.calc_total_error(expected_values, new_outputs)

		if np.array_equal(board, new_board):
			print 'board didnt change'
		else:
			print 'board changed!!!!!!!_@_@_!@_#$_!@$!#'
			#print old_error


		# Print out the error
		print 'old error: ' + str(old_error)
		print 'new error: ' + str(new_error)

		# Update the weights based on the expected values
		return new_weights

	# Calculate total error
	def calc_total_error(self, expected_values, outputs):
		return np.sum(self.total_layer_errors(expected_values, outputs))

	# Get an array containing the squared error between the expected value, and actual values
	def total_layer_errors(self, expected_values, outputs):
		val = (expected_values - outputs)
		val *= val
		#print 'val_squared ' + str(val)
		val *= .5
		#print val
		return val

	def change_error_wrt_output(self, expected_value, output):
		return -(expected_value - output)

	# Reverts the activation function so we can adjust properly
	def change_output_wrt_netinput(self, output):
		return output*(1-output)

	# How much net input changes due to weights
	# While unnecessary to define, I figured writing it would actually show its value
	def change_netinput_wrt_weight(self, hidden_output):
		return hidden_output

	def change_error_wrt_prev_hidden(self):
		change_error_wrt_output(expected_value, output)*change_output_wrt_netinput(expected_value, output)

	# How the error changes based on weight aka our delta
	# dE/dt = dE/dO * dO/dN * dN/dW
	def change_error_wrt_weight(self, weight, expected_value, output):

		return (change_error_wrt_output(expected_value, output)*
				change_output_wrt_netinput(output)*
					change_netinput_wrt_weight(weight))
	

	def backpropagate(self, inputs, weights, outputs, expected_values, learning_rate=0.1):	

		new_weights = np.zeros(weights.shape, dtype=weights.dtype)

		dels = np.zeros(outputs.shape, dtype=outputs.dtype)

		# Sets up initial error layer (the output layer)
		dels[dels.shape[0]-1,:] = self.total_layer_errors(expected_values, outputs[outputs.shape[0]-1,:])

		# The following nested loops should set up out del values for what we need to change 
		# Loop each layer
		for i in range(dels.shape[0]-1, 0, -1):

			# Get each node in this layer
			for j in range(dels.shape[1]):

				# Adds previous dels and weights to this del value (fully connected)
				for k in range(dels.shape[1]):

					# Since we are moving backwards, we use the same formula from our feed-forward
					#	algorithm and switch our j and k values.
					index = k*dels.shape[1] + j
	
					# Update our dels with the previous dels and weights
					dels[i-1, j] += dels[i,k]*weights[i,index]

				# Multiply dels by the do/di	
				dels[i-1, j] *= change_output_wrt_netinput(outputs[i-1, j])			

				# Multiply by inputs or outputs of previous layer(the inputs to this layer)
				if i == 1:
					dels[i-1, j] *= inputs[j]
				else:
					dels[i-1, j] *= outputs[i-1, j]
		# Update forward
		for i in range(dels.shape[0]):
			for j in range(dels.shape[1]):
				for k in range(dels.shape[1]):
					index = j*dels.shape[1] + k
					new_weights[i, index] = weights[i, index] + dels[i, j]




		# Return updated weights including learning rate
		return new_weights

	# Updates the weights based on the expected values and output values
	# TODO(Alex): Currently only does output layer. Extend to each layer to adjust all values and finish backpropagate.
#	def update_weights(weights, outputs, net_inputs, expected_values):
#		global learning_rate
#
#		num_layers = weights.shape[0]
#
#		# We will store our adjusted weights here
#		adjusted_weights = np.zeros(weights.shape, dtype=weights.dtype)
#
#		# Count down the layers: out->layer2->layer1
#		for i in range(num_layers, 0, -1):
#			# Elements in each layer
#			for j in range(len(outputs)):
#
#				expected_value = output[i, j];
#				output = output[i, j]
#
#				for k in range(weights.shape[1]):
#					weight_delta = change_error_wrt_weight(weights[k], 
#
#		# Start from output perceptrons and work backwards
#		for i in range(len(outputs)):
#			# Set up initial variables
#			expected_value = outputs[i]
#			output = outputs[i]
#
#			# Update each weight
#			for j in range(weights.shape[1]):
#
#				# Amount we adjust weight by, use original weights
#				weight_delta = change_error_wrt_weight(weights[j], expected_value, output)
#
#				# Haven't made any adjustments yet
#				if adjusted_weights[num_layers-1, j] == 0:
#
#					# Add new weight to adjusted weight list based on delta and learning rate
#					adjusted_weights[num_layers-1, j] += weights[num_layers-1, h] - learning_rate*weight_delta
#
#				# Make adjustments to adjusted weight
#				else:
#					# Adjust adjusted weight by delta and learning rate
#					adjusted_weights[num_layers-1, j] -= learning_rate*weight_delta
#
#
#
#		
