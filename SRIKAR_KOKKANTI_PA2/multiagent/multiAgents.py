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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

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
        if action == 'Stop':
            return -1000

        for state in newGhostStates:
            if state.getPosition() == newPos and (state.scaredTimer == 0):
                return -1000

        distancescore = currentGameState.getFood().asList();
        mindistlist = [manhattanDistance(food, newPos) for food in distancescore]
        mindist = 1+min(mindistlist,default=0)
        distancescore = newFood.asList()
        mindistlist = [manhattanDistance(food, newPos) for food in distancescore]
        currentmindist=1+min(mindistlist,default=0)

        if(currentmindist<mindist):
            return 10000;


        ghostdistlist=[manhattanDistance(ghost.getPosition(),newPos) for ghost in newGhostStates]
        ghostScore=-10*min(ghostdistlist)



        score = (1/mindist)*2000 + ghostScore

        return score

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
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"



        def minmaxhelper(gameState, agent, depth):
            if(agent>=gameState.getNumAgents()):
                agent = 0
                depth = depth+1
            if(depth==self.depth or gameState.isWin() or gameState.isLose()):
                return "",self.evaluationFunction(gameState)
            if(agent == 0):
                return maxSelector(gameState,agent,depth);
            else:
                return minSelector(gameState,agent,depth);

        def maxSelector(gameState, agent, depth):
            max=-float("inf")
            actions= gameState.getLegalActions(agent)

            if not actions or len(actions)==0:
                return "",self.evaluationFunction(gameState)
            for action in actions:
                state = gameState.generateSuccessor(agent, action)
                raction,reVal=minmaxhelper(state,agent+1,depth)
                if reVal>max:
                    max=reVal
                    reAction=action
            return reAction,max;

        def minSelector(gameState, agent, depth):
            min = float("inf")
            actions = gameState.getLegalActions(agent)
            if not actions or len(actions)==0:
                return "",self.evaluationFunction(gameState)
            for action in actions:
                state = gameState.generateSuccessor(agent, action)
                raction, reVal = minmaxhelper(state, agent+1, depth)
                if (reVal) < min:
                    min = reVal
                    reAction=action

            return reAction,min;

        k = minmaxhelper(gameState, 0, 0)

        return k[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def minmaxhelper(gameState, agent, depth,alpha,beta):
            if (agent >= gameState.getNumAgents()):
                agent = 0
                depth = depth + 1
            if (depth == self.depth or gameState.isWin() or gameState.isLose()):
                return "", self.evaluationFunction(gameState)
            if (agent == 0):
                return maxSelector(gameState, agent, depth,alpha, beta);
            else:
                return minSelector(gameState, agent, depth,alpha, beta);

        def maxSelector(gameState, agent, depth, alpha, beta):
            maxRes = -float("inf")
            actions = gameState.getLegalActions(agent)

            if not actions or len(actions) == 0:
                return "", self.evaluationFunction(gameState)
            for action in actions:
                state = gameState.generateSuccessor(agent, action)
                raction, reVal = minmaxhelper(state, agent + 1, depth, alpha,beta)
                if reVal > maxRes:
                    maxRes = reVal
                    reAction = action
                if reVal > beta:
                    return reAction,maxRes
                alpha = max(alpha,reVal)
            return reAction, maxRes;

        def minSelector(gameState, agent, depth, alpha, beta):
            minRes = float("inf")
            actions = gameState.getLegalActions(agent)
            if not actions or len(actions) == 0:
                return "", self.evaluationFunction(gameState)
            for action in actions:
                state = gameState.generateSuccessor(agent, action)
                raction, reVal = minmaxhelper(state, agent + 1, depth, alpha,beta)
                if (reVal) < minRes:
                    minRes = reVal
                    reAction = action
                if reVal<alpha:
                    return reAction,minRes
                beta=min(beta,reVal)
            return reAction, minRes;

        alpha = -(float("inf"))
        beta = float("inf")
        k = minmaxhelper(gameState, 0, 0, alpha, beta)

        return k[0]


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
        def expHelper(gameState, agent, depth):
            if(agent>=gameState.getNumAgents()):
                agent = 0
                depth = depth+1
            if(depth==self.depth or gameState.isWin() or gameState.isLose()):
                return "",self.evaluationFunction(gameState)
            if(agent == 0):
                return maxSelector(gameState,agent,depth);
            else:
                return minSelector(gameState,agent,depth);

        def maxSelector(gameState, agent, depth):
            max=-float("inf")
            actions= gameState.getLegalActions(agent)

            if not actions or len(actions)==0:
                return "",self.evaluationFunction(gameState)
            for action in actions:
                state = gameState.generateSuccessor(agent, action)
                raction,reVal=expHelper(state,agent+1,depth)
                if reVal>max:
                    max=reVal
                    reAction=action
            return reAction,max;

        def minSelector(gameState, agent, depth):
            min = 0
            actions = gameState.getLegalActions(agent)
            if not actions or len(actions)==0:
                return "",self.evaluationFunction(gameState)
            prob=1/len(actions)
            for action in actions:
                state = gameState.generateSuccessor(agent, action)
                raction, reVal = expHelper(state, agent+1, depth)
                min=min+(prob*reVal);
                reAction=action


            return reAction,min;

        k = expHelper(gameState, 0, 0)

        return k[0]



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: weighted average of food,ghosts and capsules
    """
    "*** YOUR CODE HERE ***"


    score=0
    distancescore = currentGameState.getFood().asList();
    mindistlist = [manhattanDistance(food, currentGameState.getPacmanPosition()) for food in distancescore]

    mindist = min(mindistlist, default=0)
    if(mindist==0):
        return currentGameState.getScore()
    score=score+(1/mindist) *10

    distancescore = currentGameState.getGhostPositions();
    mindistlist = [manhattanDistance(ghost, currentGameState.getPacmanPosition()) for ghost in distancescore]
    mindist = min(mindistlist, default=1)

    if(mindist!=0):
        score = score - (1 / mindist) * 10

    distancescore = currentGameState.getCapsules();
    mindistlist = [manhattanDistance(capsule, currentGameState.getPacmanPosition()) for capsule in distancescore]
    mindist = min(mindistlist, default=1)

    if (mindist != 0):
        score = score - (1 / mindist) * 3



    return score+ currentGameState.getScore();



# Abbreviation
better = betterEvaluationFunction
