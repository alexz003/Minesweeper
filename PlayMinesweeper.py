import Minesweeper
import NeuralNetwork


def main():
	game = Minesweeper.Minesweeper()

	NN = NeuralNetwork.NeuralNetwork(game)
	NN.create_model(game.N*game.N, game.N*game.N, num_layers=1)
	NN.run_all_trials(game)
main()
