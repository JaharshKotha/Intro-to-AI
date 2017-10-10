
# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        #print legalMoves

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #print legalMoves

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        penalty = 10.0
        return_value = successorGameState.getScore()
        g_val = 0
        pos = currentGameState.getPacmanPosition()
        foodList = newFood.asList()
        min_dis = 999999
        min_gdis = 999999
        upper_limit = 10000

        cur_food_list = currentGameState.getFood().asList()



        for itr in cur_food_list:
            dist1 = abs(itr[0] -pos[0])
            dist2 = abs(itr[1] - pos[1])
            tmp =  dist1+dist2
            if (min_dis > tmp):
                min_dis = tmp
                food_corr = itr

           # print nearestFood

        push_dist = abs(food_corr[0] -newPos[0]) + abs(food_corr[1] - newPos[1])

        if currentGameState.getScore() < successorGameState.getScore():
            return_value += 5



        for ghost in newGhostStates:
            td1 = abs(newPos[0] - ghost.getPosition()[0])
            td2 = abs(newPos[1] - ghost.getPosition()[1])
            if ((td1+td2)<= 2):
                val_func = (3 - (td1+td2))
                g_val -= val_func*val_func
                # print ghostEval
                # print "this"
                # print ghostEval**2


        return_value = g_val - push_dist
        return return_value

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)




class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):


        allowed_moves = gameState.getLegalActions(0)
        #print gameState.getLegalActions(0)
        #print gameState.getNumAgents()
        val = []
        for action in allowed_moves:
            next_state = gameState.generateSuccessor(0, action)
            val += [self.minimizer(0,next_state, 1)]

        check = max(val)
        pos=[]
        i=0
        while(i<len(val)):
            if val[i] == check:
                pos+=[i]
            i+=1

        return allowed_moves[pos[0]]


    def check_stop(self,cd,gs):
        depth_reach = True if self.depth==cd else False
        loss = True if gs.isLose() else False
        win = True if gs.isWin() else False
        if(depth_reach or loss or win):
            return True
        else:
            return False


    def minimizer(self, Depth, gameState, level):
        stop = self.check_stop(Depth, gameState)
        if (stop):
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(level)
        resultStates=[]
        for action in legalMoves:
            resultStates+=[gameState.generateSuccessor(level, action)]
        scores=[]
        if (level >= gameState.getNumAgents() - 1):
             Depth+=1
             for state in resultStates:
                stop = self.check_stop(Depth, state)
                if (stop):
                    scores+=[self.evaluationFunction(state)]
                    continue
                legalMoves = state.getLegalActions(0)
                ns=[]
                temp = []
                for move in legalMoves:
                    temp+=[self.minimizer(Depth, state.generateSuccessor(0, move), 1)]
                scores+=[max(temp)]
                #scores+=[self.maximizer(Depth + 1, state)]
        else:
            for action in legalMoves:
                st = gameState.generateSuccessor(level, action)
                scores+= [self.minimizer(Depth,st, level + 1)]

        return min(scores)


class AlphaBetaAgent(MultiAgentSearchAgent):

    def reset(self, state, depth,tree_level):
        r_val = True if tree_level == state.getNumAgents() else False
        if r_val:
            return[depth+1,0,True]
        else:
            return [depth,tree_level,False]




    def getAction(self, gameState):

        def prunning(state, depth, tree_level, alpha, beta):

            if self.reset(state,depth,tree_level)[2]:
                depth = self.reset(state,depth,tree_level)[0]
                tree_level = self.reset(state,depth,tree_level)[1]

            stop = None
            depth_reach = True if self.depth == depth else False
            loss = True if state.isLose() else False
            win = True if state.isWin() else False
            if (depth_reach or loss or win):
                stop =True
            else:
                stop=False

            if stop:
                return_val =[self.evaluationFunction(state), None]
                return return_val

            if (tree_level==0):
                return max_value(state, depth, tree_level, alpha, beta)
            else:
                return min_value(state, depth, tree_level, alpha, beta)

        def max_value(state, depth, tree_level, A, B):
            v = float('-inf')
            a = None
            legal_moves = state.getLegalActions(tree_level)
            for action in legal_moves:
                successor = state.generateSuccessor(tree_level, action)
                score,_ = prunning(successor, depth, tree_level + 1, A, B)
                v, a = max((v, a), (score, action))
                if v > B:
                    return [v, a]
                A = max(A, v)


            return [v, a]

        def min_value(state, depth, tree_level, A, B):
            bestScore = float('inf')
            bestAction = None

            for action in state.getLegalActions(tree_level):
                successor = state.generateSuccessor(tree_level, action)
                score, _ = prunning(successor, depth, tree_level + 1, A, B)
                bestScore, bestAction = min((bestScore, bestAction), (score, action))
                if bestScore < A:
                    return [bestScore, bestAction]
                B = min(B, bestScore)

            return [bestScore, bestAction]

        alpha = float("-inf")
        beta = float("inf")

        action = prunning(gameState, 0, 0,alpha,beta)
        return action[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

