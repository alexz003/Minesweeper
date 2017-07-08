import numpy as np

class NeuralNetwork():
			
	# Sigmoid with vectors
	def sigmoid(x):
		return 1/(1 + np.exp(x))

	# Sigmoid derivative with vectors
	def div_sigmoid(x):
		return sigmoid(x)*(1-sigmoid(x))

	# Create the neural net model
	def create_model(num_layers = 1, nodes_per_layer, num_trials=10000):

		# NxM weights matrix with random initial values
		weights = np.random.random_uniform((num_layers, nodes_per_layer))
		
		# Output matrix
		output = np.zeros(num_inputs)

		# Run each trial
		for i in range(num_trials):
			# Set weights based on trial
			weights = run_trial(None, np.array(0), weights, output)

	# Run a single trial of Neural Network
	def run_trial(game, inputs, weights):

		# Make new game if not created or not running
		if game == None or not game.getStatus() == 0:
			game = Minesweeper.Minesweeper()
			game.start_game()	
	
		# Train this test case	
		weights = train(game, weights)	 
		
	# Feedforward on weights with board and return outputs
	def test(game, weights):
	
		# Get current board state	
		board = game.getBoard().flatten()
	
		# TODO(Alex): Currently only outputs total out. To save future computation,
		#		we can save the net input and output of each layer and output
		#		those arrays instead
		

		# Create net inputs and outputs containing the results of each layer
		net_inputs = np.ones(weights.shape)
		outputs = np.ones(weights.shape)
	
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

			# Add non-sigmoid input values
			net_inputs[i,:] = np.mult(num_previous, temp_weights)

			# Add sigmoid output values
			outputs[i,:] = sigmoid(np.mult(sum_previous, temp_weights))

		# Return the outputs
		return outputs, net_inputs

	def train(game, weights):

		# Get the new game board state
		board = game.getBoard().flatten()

		# Feed forward for output values
		outputs, net_inputs = test(game, board.flatten(), weights)

		# Get the best output index
		index = 0
		for i in range(outputs.shape[1]):
			if outputs[outputs.shape[0]-1, i] > outputs[outputs.shape[0]-1, index]:
				index = i;

		# Gets location of move based on inputs
		x = math.floor(i/game.N)
		y = i % game.N

		# Sends move to the game and retrieves new board state
		game.send_move(x, y)
		new_board = game.getBoard().flatten()

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

		# Get new weights after backpropagating
		new_weights = backpropagate(inputs, weights, outputs, expected_values)

		# Calculate weight differences
		old_error = calc_total_error(expected_values, outputs)
		new_outputs = test(game, new_weights)
		new_error = calc_total_error(expected_values, new_outputs)

		# New error should always be lower than the old error
		if new_error > old_error:
			print 'You fucked up somewhere...\n'

		# Update the weights based on the expected values
		return new_weights

	# Calculate total error
	def calc_total_error(expected_values, outputs):
		return 1/2*np.sum(np.power(expected_values - outputs, 2))

	# Get an array containing the squared error between the expected value, and actual values
	def total_layer_errors(expected_values, outputs):
		return 1/2*np.power(expected_values - outputs, 2)

	def change_error_wrt_output(expected_value, output):
		return -(expected_value - output)

	# Reverts the activation function so we can adjust properly
	def change_output_wrt_netinput(output):
		return output*(1-output)

	# How much net input changes due to weights
	# While unnecessary to define, I figured writing it would actually show its value
	def change_netinput_wrt_weight(hidden_output):
		return hidden_output

	def change_error_wrt_prev_hidden():
		change_error_wrt_output(expected_value, output)*change_output_wrt_netinput(expected_value, output)

	# How the error changes based on weight aka our delta
	# dE/dt = dE/dO * dO/dN * dN/dW
	def change_error_wrt_weight(weight, expected_value, output):

		return (change_error_wrt_output(expected_value, output)*
				change_output_wrt_netinput(output)*
					change_netinput_wrt_weight(weight))
	

	def backpropagate(inputs, weights, outputs, expected_values, learning_rate=0.1):	

		new_weights = np.zeros(weights.shape, dtype=np.int32)

		dels = np.zeros(outputs.shape, dtype=outputs.dtype)
		dels[dels.shape[0]-1,:] = total_layer_errors(expected_values, outputs[outputs.shape[0]-1,:])


		# The following nested loops should set up out del values for what we need to change 
		# Loop each layer
		for i in range(dels.shape[0]-1, 0, -1):

			# Get each node in this layer
			for j in range(dels.shape[1]-1):

				# Adds previous dels and weights to this del value (fully connected)
				for k in range(dels.shape[1]-1):
					dels[i-1, j] += dels[i,k]*weights[i,k]

				# Multiply out dels by the do/di	
				dels[i-1, j] *= change_output_wrt_netinput(outputs[i-1, j])			

				# Multiply by inputs or outputs of previous layer(the inputs to this layer)
				if i == 1:
					dels[i-1, j] *= inputs[j]
				else:
					dels[i-1, j] *= outputs[i-1, j]

		# Return updated weights including learning rate
		return weights + learning_rate*dels

	# Updates the weights based on the expected values and output values
	# TODO(Alex): Currently only does output layer. Extend to each layer to adjust all values and finish backpropagate.
	def update_weights(weights, outputs, net_inputs, expected_values):
		global learning_rate

		num_layers = weights.shape[0]

		# We will store our adjusted weights here
		adjusted_weights = np.zeros(weights.shape, dtype=weights.dtype)

		# Count down the layers: out->layer2->layer1
		for i in range(num_layers, 0, -1):
			# Elements in each layer
			for j in range(len(outputs)):

				expected_value = output[i, j];
				output = output[i, j]

				for k in range(weights.shape[1]):
					weight_delta = change_error_wrt_weight(weights[k], 

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



		
