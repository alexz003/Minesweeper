import numpy as np
from random import randint

class Minesweeper(object):

	# Sets size of board
	N = 20
	board = None

	# Bombs
	bombCount = 15
	bombs = None

	# Game Data
	gameStarted = False

	# Flood fill data
	dx = np.array([(-1, -1), (-1, 0), (-1, 1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)], dtype=tuple)
	
	def __init__(self):
		self.N = 20
		self.board = None
		self.bombCount = 15
		self.bombs = None
		self.gameStarted = False	
		self.dx = np.array([(-1, -1), (-1, 0), (-1, 1), (0, 1), (0, -1), (1, -1), (1, 0), (1, 1)], dtype=tuple)
	
	def startGame(self):

		self.board = self.createboard()
		status = self.loop()
		if status == -1:
			print "Sorry you lose."
		else:
			print "You won!."

	# Creates NxN board
	# Values:
	#	-2 = bomb
	#	-1 = unvisited
	#	 x = number of adjacent bombs
	#
	def createboard(self):
		# Create an empty self.Nxself.N self.board
		temp = np.zeros([self.N, self.N], dtype=np.int32)
		temp.fill(-1)
		return temp

	# Check if a bomb is in this location
	def checkBomb(self, loc):

		for i in range(len(self.bombs)):
			if self.bombs[i] == loc:
				return True
		return False

	# Creates a single bomb, avoids duplicates
	def createBomb(self, loc):

		# Create a location tuple for the location of the bomb
		temp = (randint(0, self.N-1), randint(0, self.N-1))
		
		# Recreate bomb if location is not unique
		# Avoids first choice by loc being (-1, -1) when not first
		while self.checkBomb(temp) or temp == loc:
			temp = (randint(0, self.N-1), randint(0, self.N-1))
		
		return temp;
		
	# Create list of bombs, avoiding first location and duplicates
	def createbombs(self, loc):

		if self.bombs == None:
			self.bombs = np.empty([self.bombCount], dtype=tuple)
			self.bombs.fill((-1, -1))

		# Creates bombs and adds to bomb-list
		for i in range(self.bombCount):
			self.bombs[i] = self.createBomb(loc)

	# Prints out the current state of the board
	def printBoard(self):
		
		for i in range(self.N):
			for j in range(self.N):
				loc = (j, i)
				print("%2d" % (self.board[loc])),
			print ""

	# Gets the number of adjacent bombs at a location
	def getAdjacentbombs(self, loc):
		count = 0
		for i in range(len(self.dx)):
			newLoc = tuple(loc + self.dx[i])
			if self.checkBomb(newLoc):
				count += 1
		return count

	# Flood fill the values of adjacent bombs
	def floodFill(self, loc):

		# Check out of bounds
		if loc[0] < 0 or loc[0] >= self.N or loc[1] < 0 or loc[1] >= self.N:
			return

		# Check for self.bombs
		if self.checkBomb(loc):
			return 

		# Unvisited location
		if self.board[loc] == -1:
			# Change board value to number of adjacent bombs
			self.board[loc] = self.getAdjacentbombs(loc)
			if self.board[loc] == 0:
				# If no adjacent bombs, flood fill
				for i in range(len(self.dx)):
					self.floodFill(tuple(loc + self.dx[i]))

	# Choose a location
	def chooseLocation(self, loc):
	 
		# Creates bombs after game has started
		if not self.gameStarted:
			self.createbombs(loc)
			self.gameStarted = True

		# If a bomb is here, show it	
		if self.checkBomb(loc):
			self.board[loc] = -2
			return

		# Flood fill from location
		self.floodFill(loc)

	# Returns the status of the game:
	# 	-1: Loss
	#	 0: Running
	#	 1: Win
	def gameStatus(self):
		count = 0
		
		for i in range(self.N):
			for j in range(self.N):
				# Bomb found
				if self.board[i,j] == -2:
					return -1

				# Unvisited location
				if self.board[i,j] == -1:
					count += 1

		# Check if the only empty spots left are self.bombs
		if count == len(self.bombs):
			return 1
		
		# Game still in progress
		return 0

	# Returns the current board
	def getBoard(self):
		return self.board

	# Main game loop, while the game is still running
	def loop(self):	
		# Start game with a random first location
		loc = (randint(0, self.N-1), randint(0, self.N-1))
		self.chooseLocation(loc)
		self.printBoard()

		# Get next location
		# TODO: Change to take the output from learning model
		status = 0
		while status == 0:
			x = raw_input("x: ")
			y = raw_input("y: ")
			loc = (int(x), int(y))
			self.chooseLocation(loc)
			self.printBoard()
			self.status = self.gameStatus()
		
		return status
