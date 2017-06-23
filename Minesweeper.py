import numpy as np
from random import randint

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


# Creates NxN board
# Values:
#	-2 = bomb
#	-1 = unvisited
#	 x = number of adjacent bombs
#
def createBoard():
	# Create an empty NxN board
	temp = np.zeros([N, N], dtype=np.int32)
	temp.fill(-1)
	return temp

# Check if a bomb is in this location
def checkBomb(loc):
	global bombs

	for i in range(len(bombs)):
		if bombs[i] == loc:
			return True
	return False

# Creates a single bomb, avoids duplicates
def createBomb(loc):

	# Create a location tuple for the location of the bomb
	temp = (randint(0, N-1), randint(0, N-1))
	
	# Recreate bomb if location is not unique
	# Avoids first choice by loc being (-1, -1) when not first
	while checkBomb(temp) or temp == loc:
		temp = (randint(0, N-1), randint(0, N-1))
	
	return temp;
	
# Create list of bombs, avoiding first location and duplicates
def createBombs(loc):
	global bombs

	if bombs == None:
		bombs = np.empty([bombCount], dtype=tuple)
		bombs.fill((-1, -1))

	# Creates bombs and adds to bomb-list
	for i in range(bombCount):
		bombs[i] = createBomb(loc)

# Prints out the current state of the board
def printBoard():
	global board

	for i in range(N):
		for j in range(N):
			loc = (j, i)
			print("%2d" % (board[loc])),
		print ""

# Gets the number of adjacent bombs at a location
def getAdjacentBombs(loc):
	global dx
	count = 0
	for i in range(len(dx)):
		newLoc = tuple(loc + dx[i])
		if checkBomb(newLoc):
			count += 1
	return count

# Flood fill the values of adjacent bombs
def floodFill(loc):
	global board

	# Check out of bounds
	if loc[0] < 0 or loc[0] >= N or loc[1] < 0 or loc[1] >= N:
		return

	# Check for bombs
	if checkBomb(loc):
		return 

	# Unvisited location
	if board[loc] == -1:
		# Change board value to number of adjacent bombs
		board[loc] = getAdjacentBombs(loc)
		if board[loc] == 0:
			# If no adjacent bombs, flood fill
			for i in range(len(dx)):
				floodFill(tuple(loc + dx[i]))

# Choose a location
def chooseLocation(loc):
	global gameStarted
 
	# Creates bombs after game has started
	if not gameStarted:
		createBombs(loc)
		gameStarted = True

	# If a bomb is here, show it	
	if checkBomb(loc):
		board[loc] = -2
		return

	# Flood fill from location
	floodFill(loc)

# Returns the status of the game:
# 	-1: Loss
#	 0: Running
#	 1: Win
def gameStatus():
	global board
	count = 0
	
	for i in range(N):
		for j in range(N):
			# Bomb found
			if board[i,j] == -2:
				return -1

			# Unvisited location
			if board[i,j] == -1:
				count += 1

	# Check if the only empty spots left are bombs
	if count == len(bombs):
		return 1
	
	# Game still in progress
	return 0


# Main game loop, while the game is still running
def loop():	
	# Start game with a random first location
	loc = (randint(0, N-1), randint(0, N-1))
	chooseLocation(loc)
	printBoard()

	# Get next location
	# TODO: Change to take the output from learning model
	status = 0
	while status == 0:
		x = raw_input("x: ")
		y = raw_input("y: ")
		loc = (int(x), int(y))
		chooseLocation(loc)
		printBoard()
		status = gameStatus()
	
	return status

def main():
	global board

	board = createBoard()
	status = loop()
	if status == -1:
		print "Sorry you lose."
	else:
		print "You won!"


main()
