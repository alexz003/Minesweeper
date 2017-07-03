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
		outputs_ = np.ones(len(board))

		# Weights are NxM while board is 1xM
		num_layers = weights.shape
		
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
			outputs_ = np.mult(num_previous, temp_weights)

		# Return the outputs
		return outputs, outputs_

	def train(game, weights):

		# Get the new game board state
		board = game.getBoard().flatten()

		# Feed forward for output values
		outputs, outputs_ = test(game, board.flatten(), weights)

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
		return update_weights(weights, outputs, outputs_, expected_value)


	def update_weights(weights, outputs, outputs_, expected_values):
		global learning_rate

		error = learning_rate*(expected_values - outputs_)

		outputs += outputs*error

		

		

