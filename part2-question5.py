class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        "*** YOUR CODE HERE ***"
        self.totalFoodCost = "FFFF" #This value holds where the food is when converted to binary.

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        return (self.startingPosition, self.totalFoodCost)     # State is a tuple which holds the position and the totalFoodCost
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        x, y = state
        if y == "TTTT":            #If state is zero then the binary value has all zeros which implies no food is left out.
            return True
        else:
            return False
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            position, foodValue = state
            x, y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextPosition = (nextx, nexty)
                tempFoodValue = foodValue
                if (nextx, nexty) in self.corners:
                    b = self.corners.index((nextx,nexty))
                    if b == 0:
                        tempFoodValue = tempFoodValue[:0] + "T" + tempFoodValue[0 + 1:]
                    if b == 1:
                        tempFoodValue = tempFoodValue[:1] + "T" + tempFoodValue[1 + 1:]
                    if b == 2:
                        tempFoodValue = tempFoodValue[:2] + "T" + tempFoodValue[2 + 1:]
                    if b == 3:
                        tempFoodValue = tempFoodValue[:3] + "T" + tempFoodValue[3 + 1:]
                nextState = (nextPosition, tempFoodValue)
                successors.append( ( nextState, action, 1) )
        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)
