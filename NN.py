import numpy as np

class NeuralNetwork():
			
	learning_rate = 0.1

	# Sigmoid with vectors
	def sigmoid(x):
		return 1/(1 + np.exp(x))

	# Sigmoid derivative with vectors
	def div_sigmoid(x):
		return sigmoid(x)*(1-sigmoid(x))

	# Create the neural net model
	def create_model(num_layers = 1, nodes_per_layer, num_trials=10000):

		# NxM weights matrix with random initial values
		weights = np.random.random_sample((num_layers, nodes_per_layer))
		
		# Output matrix
		output = np.zeros(num_inputs)

		# Run each trial
		for i in range(num_trials):
			# Set weights based on trial
			weights = run_trial(None, np.array(0), weights, output)

	# Run a single trial of Neural Network
	def run_trial(game, inputs, weights):

		# Make new game if old or non-eistatnt
		if game == None or not game.getStatus() == 0:
			game = Minesweeper.Minesweeper()	
	
		# Train this test case	
		weights = train(game, weights)	 
		
	# Feedforward on weights with board and return outputs
	def test(game, weights):
	
		# Get current board state	
		board = game.getBoard().flatten()

		# Create outputs with 1's so we can multiply by initial values
		outputs = np.ones(len(board))
		net_inputs = np.ones(len(board))

		# Weights are NxM while board is 1xM
		num_layers = weights.shape[0]
		
		# Go through each layer
		for i in range(num_layers):

			# Get's the weights in each layer
			temp_weights = weights[i,:]
			
			# Total sum of the previous layer
			sum_previous = 0
			
			# First layer has sum_previous be the sum of inputs
			if i == 0:
				sum_previous = np.add(board)

			# Set sum_previous to sum of previous row's outputs
			else:
				sum_previous = np.add(outputs)

			# Multiply current weight by sum_previous, then take the sigmoid
			outputs = sigmoid(np.mult(sum_previous, temp_weights))

			# Non-sigmoid output values
			net_inputs = np.mult(num_previous, temp_weights)

		# Return the outputs
		return outputs, net_inputs

	def train(game, weights):

		# Get the new game board state
		board = game.getBoard().flatten()

		# Feed forward for output values
		outputs, net_inputs = test(game, board.flatten(), weights)

		# TODO(Alex): Send best output to game as selection

		# Set up vector of expected values
		expected_values = np.zeros(len(weights), dtype = float)

		# Supervised Learning - expected values of weights
		# Good move:	1 - Board changes and not a bomb
		# Bad move:	0 - Bomb found
		for i in range(len(board)):
			if not board[i] == new_board[i] and not game.gameStatus() == -1
				expected_values[i] = 1
			if newBoard[i] == -2
				expected_values[i] = 0

		# Update the weights based on the expected values
		return update_weights(weights, outputs, net_inputs, expected_value)

	# Get an array containing the squared error between the expected value, and actual values
	def total_layer_error(expected_values, outputs):
		return 1/2*np.sum(np.power(expected_value - output, 2))

	def change_error_wrt_output(expected_value, output):
		return -(expected_value - output)

	# Reverts the activation function so we can adjust properly
	def change_output_wrt_netinput(output):
		return outputs*(1-output)

	# How much net input changes due to weights
	# While unnecessary to define, I figured writing it would actually show its value
	def change_netinput_wrt_weight(hidden_output):
		return hidden_output

	# How the error changes based on weight aka our delta
	# dE/dt = dE/dO * dO/dN * dN/dW
	def change_error_wrt_weight(weight, expected_value, output):

		return (change_error_wrt_output(expected_value, output)*
				change_output_wrt_netinput(output)*
					change_netinput_wrt_weight(weight))
	
	# Updates the weights based on the expected values and output values
	# TODO(Alex): Currently only does output layer. Extend to each layer to adjust all values and finish backpropagate.
	def update_weights(weights, outputs, net_inputs, expected_values):
		global learning_rate

		num_layers = weights.shape[0]

		# We will store our adjusted weights here
		adjusted_weights = np.zeros(weights.shape, dtype=weights.dtype)

		# Start from output perceptrons and work backwards
		for i in range(len(outputs)):
			# Set up initial variables
			expected_value = outputs[i]
			output = outputs[i]

			# Update each weight
			for j in range(weights.shape[1]):

				# Amount we adjust weight by, use original weights
				weight_delta = change_error_wrt_weight(weights[j], expected_value, output)

				# Haven't made any adjustments yet
				if adjusted_weights[num_layers-1, j] == 0:

					# Add new weight to adjusted weight list based on delta and learning rate
					adjusted_weights[num_layers-1, j] += weights[num_layers-1, h] - learning_rate*weight_delta

				# Make adjustments to adjusted weight
				else:
					# Adjust adjusted weight by delta and learning rate
					adjusted_weights[num_layers-1, j] -= learning_rate*weight_delta



		
