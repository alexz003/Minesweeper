from __future__ import division
from graphics import *
from random import randint


win = None
score = 10
exploded = False
bombs = []
dx = [-1, -1, -1, 0, 0, 1, 1, 1]
dy = [-1, 0, 1, 1, -1, -1, 0, 1]

topLeft = Point(5, 55)
lowerRight = Point(205, 255)


# Creates game board
def createBoard():
    border = Rectangle(topLeft, lowerRight)
    border.draw(win)

    for val in range(10):
        line1 = Line(Point(topLeft.x + val*20, topLeft.y), Point(topLeft.x + val*20, lowerRight.y))
        line2 = Line(Point(topLeft.x, topLeft.y + val*20), Point(lowerRight.x, topLeft.y + val*20))
        line1.draw(win)
        line2.draw(win) 

# Check if duplicate bomb
def checkBomb(loc):
    for b in bombs:
        if b.x == loc.x and b.y == loc.y:
            return 1

    return 0

def displayBombs():
    for bomb in bombs:
        circle = Circle(Point(topLeft.x + bomb.x*20 + 10, topLeft.y + bomb.y*20 + 10), 5)
        circle.setFill('grey')
        circle.draw(win)

# Creates bombs
def createBombs():
    numBombs = 10
    global bombs
    for i in range(numBombs):
            temp = None
            # Loops new locations until new bomb location is created
            while temp == None:
                temp = Point(randint(0, 9), randint(0, 9))
                if not checkBomb(temp):
                    bombs.append(temp)
                    print 'Bomb created at (' + str(temp.x) + ', ' + str(temp.y) + ')'
                    continue
                temp = None
                
def showValue(cellLoc, value):
    loc = Point(topLeft.x + cellLoc.x*20 + 10, topLeft.y + cellLoc.y*20 + 10)
    text = Text(loc, str(value))
    text.draw(win)

# Creates the text area
def createTextArea():
    topLeft = Point(30, 10)
    text = Text(topLeft, 'Score: ' + str(10 - score))
    text.draw(win)

# Checks if game is finished
def gameFinished():
    return score == 0 or exploded == True

# Gets the x and y value of the cell selected within the grid
def cellLoc(loc):
    # Area clicked not within grid
    if loc.x < 5 or loc.x > 205 or loc.y < 55 or loc.y > 255:
        return Point(-1, -1)
    
    x = loc.x - 5
    y = loc.y - 55

    cellX = x//20
    cellY = y//20
    
    print 'Checking cell (' + str(cellX) + ', ' + str(cellY) + ')'
    return Point(cellX, cellY)

def adjacentBombs(cellLoc):
    sum = 0    
    for i in range(len(dx)):
        if checkBomb(Point(cellLoc.x + dx[i], cellLoc.y + dy[i])):
            sum += 1

    return sum

# Fills in 0's for all adjacent zeros            
def floodFill(cellLoc, checked):

    #print 'Checking location (' + str(cellLoc.x) + ', ' + str(cellLoc.y) + ')'
    for check in checked:
        if check.x == cellLoc.x and check.y == cellLoc.y:
            return

    if cellLoc.x < 0 or cellLoc.x > 9 or cellLoc.y < 0 or cellLoc.y > 9:
        return

    adj = adjacentBombs(cellLoc)
    if adj == -1:
        return
    if adj == 0:
        showValue(cellLoc, adj)
        checked.append(cellLoc)
        for i in range(len(dx)):
            floodFill(Point(cellLoc.x + dx[i], cellLoc.y + dy[i]), checked)
    if adj > 0:
        checked.append(cellLoc)
        showValue(cellLoc, adj)

# Checks the cell for bomb(-1) or number of adjacent bombs
def checkCell(cellLoc):
        if checkBomb(cellLoc):
            return -1
        return adjacentBombs(cellLoc)

# Main loop that runs the game
def loop():
    global exploded
    while not exploded:
        mouseLoc = win.getMouse()
        loc = cellLoc(mouseLoc)
        
        #
        if loc.x != -1 and loc.y != -1:
            cellType = checkCell(loc)
            if cellType == -1:
                exploded = True
                displayBombs()
                print 'exploded: ' + str(exploded)
            else:
                floodFill(loc, [])
            
            
        


def main():
    global win
    win = GraphWin('Minesweeper', 210, 260)
    
    createTextArea()    
    createBoard()
    createBombs()
    loop()

    win.getMouse()
    win.close()

main()
