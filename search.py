# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    import util
    stack = util.Stack()
    start_state = problem.getStartState()
    if (problem.isGoalState(start_state)):
        return None
    flg = 0
    res = []
    visited = set()
    visited.add(start_state)
    for i in problem.getSuccessors(start_state):
        tem = [i[0], i[1]]
        stack.push(tem)

    while not stack.isEmpty():
        t = stack.pop()
        visited.add(t[0])

        if (problem.isGoalState(t[0])):
            res = t[1]
            flg = 1
            break
        else:

            for j in problem.getSuccessors(t[0]):
                if j[0] not in visited:
                    ltem = t[1] + "," + j[1]
                    stack.push([j[0], ltem])

    final = []
    if flg == 1:
        data = res.split(",")
        for temp in data:
            final.append(temp)
        return final
    else:
        return None


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    import util
    queue = util.Queue()
    start_state = problem.getStartState()
    if (problem.isGoalState(start_state)):
        return None
    flg = 0
    res = []
    visited = set()
    visited.add(start_state)
    many_paths = set()
    many_paths.add(start_state)
    for i in problem.getSuccessors(start_state):
        tem = [i[0], i[1]]
        queue.push(tem)
        many_paths.add(i[0])

    while not queue.isEmpty():
        t = queue.pop()
        visited.add(t[0])

        if (problem.isGoalState(t[0])):
            res = t[1]
            flg = 1
            break
        else:

            for j in problem.getSuccessors(t[0]):
                if j[0] not in visited and j[0] not in many_paths:
                    ltem = t[1] + "," + j[1]
                    queue.push([j[0], ltem])
                    many_paths.add(j[0])

    final = []
    if flg == 1:
        data = res.split(",")
        for temp in data:
            final.append(temp)
        return final
    else:
        return None



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    import util
    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    if (problem.isGoalState(start_state)):
        return None
    flg = 0
    res = []
    visited = set()
    visited.add(start_state)
    many_paths = set()
    many_paths.add(start_state)
    for i in problem.getSuccessors(start_state):
        tem = [i[0],i[1],i[2]]
        priority_queue.push(tem,i[2])
        many_paths.add(i[0])


    while not priority_queue.isEmpty():
        if flg==1:
            break
        t = priority_queue.pop()
        if t[0] in visited:
            t = priority_queue.pop()
        visited.add(t[0])

        if (problem.isGoalState(t[0])):
            res = t[1]
            flg = 1
            break
        else:
            #print "inside succ"
            for j in problem.getSuccessors(t[0]):
                if j[0] not in visited :
                    ltem = t[1] + "," + j[1]
                    cost = t[2]+j[2]
                    #print j
                    priority_queue.push([j[0], ltem,cost],cost)
                    many_paths.add(j[0])

    final = []
    if flg == 1:
        data = res.split(",")
        for temp in data:
            final.append(temp)
        return final
    else:
        return None


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    import util
    priority_queue = util.PriorityQueue()
    start_state = problem.getStartState()
    if (problem.isGoalState(start_state)):
        return None
    flg = 0
    res = []
    visited = set()
    visited.add(start_state)
    many_paths = set()
    many_paths.add(start_state)
    for i in problem.getSuccessors(start_state):
        tem = [i[0], i[1], i[2]]
        initial_cost = i[2] + heuristic(i[0], problem)
        priority_queue.push(tem, initial_cost)
        many_paths.add(i[0])

    while not priority_queue.isEmpty():
        t = priority_queue.pop()
        if t[0] in visited:
            continue
        visited.add(t[0])

        if (problem.isGoalState(t[0])):
            res = t[1]
            flg = 1
            break
        else:
            for j in problem.getSuccessors(t[0]):
                if j[0] not in visited:
                    ltem = t[1] + "," + j[1]
                    cost = heuristic(j[0], problem) + j[2] + t[2]
                    temp_cost = j[2]+t[2]
                    priority_queue.push([j[0], ltem, temp_cost], cost)
                    many_paths.add(j[0])
    print len(visited)
    final = []
    if flg == 1:
        data = res.split(",")
        for temp in data:
            final.append(temp)
        return final
    else:
        return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
