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
        """Finding the distance to food"""
        newFoodlist =  util.matrixAsList(newFood.data)
        min_food_distance = 1
        closeFoodDist = 999999
        farFoodDist = 1
        
        for food in newFoodlist:
            dist = util.manhattanDistance(newPos, food)
            min_food_distance += dist
            if dist < closeFoodDist:
                closeFoodDist = dist
            if dist > farFoodDist:
                farFoodDist = dist
                
        # Distance metric to ghosts  
        dist_to_ghosts = 1 # init to 1 to prevent div by 0
        scaredGhost = 10
        close_to_ghosts = 0
        
        for ghostIdx in range(len(newGhostStates)):
            ghostState = newGhostStates[ghostIdx]
            scaredTime = newScaredTimes[ghostIdx]
                        
            if scaredTime > 0: 
                scaredGhost += 50
                continue
            else:
                ghostDist = util.manhattanDistance(newPos, ghostState.configuration.pos)
                dist_to_ghosts += ghostDist
                if ghostDist <= 1:
                    close_to_ghosts -= 200
        
        return  successorGameState.getScore() + (1/float(closeFoodDist)) - (1/float(min_food_distance)) + close_to_ghosts + scaredGhost
    
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
        """
        "*** YOUR CODE HERE ***"
        
        def minimax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            if agent == 0:  # maximize 
                return max(minimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else:  # minimze 
                newAgent = agent + 1  # calculate the next agent and increase depth accordingly.
                if gameState.getNumAgents() == newAgent:
                    nextAgent = 0
                if nextAgent == 0:
                   depth += 1
                return min(minimax(newAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))

        """maximize action for the root node"""
        maximum = -float("inf")
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            maxV = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            if maxV > maximum or maximum == -float("inf"):
                maximum = maxV
                action = agentState

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def maxvalue(agent, depth, game_state, a, b):  # maximum function
            v = -float("inf")
            for newState in game_state.getLegalActions(agent):
                v = max(v, alphabetaprune(1, depth, game_state.generateSuccessor(agent, newState), a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v

        def minvalue(agent, depth, game_state, a, b):  # minimum function
            v = float("inf")

            next_agent = agent + 1 
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:
                depth += 1

            for newState in game_state.getLegalActions(agent):
                v = min(v, alphabetaprune(next_agent, depth, game_state.generateSuccessor(agent, newState), a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v

        def alphabetaprune(agent, depth, game_state, a, b):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:
                return self.evaluationFunction(game_state)

            if agent == 0:  # maximize
                return maxvalue(agent, depth, game_state, a, b)
            else:  # minimize 
                return minvalue(agent, depth, game_state, a, b)

        """Performing max function to the root node"""
        maxV = float("-inf")
        bestaction = Directions.WEST
        alpha = -float("inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):
            ghostValue = alphabetaprune(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if ghostValue > maxV:
                maxV = ghostValue
                bestaction = action
            if maxV > beta:
                return maxV
            alpha = max(alpha, maxV)

        return bestaction


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
        def expectimax(agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth: 
                return self.evaluationFunction(gameState)
            if agent == 0: 
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            else:  # performing expectimax action for ghosts/chance nodes.
                newAgent = agent + 1  # calculate the next agent and increase depth accordingly.
                if gameState.getNumAgents() == newAgent:
                    newAgent = 0
                if newAgent == 0:
                    depth += 1
                expect = sum(expectimax(newAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent)) / float(len(gameState.getLegalActions(agent)))
                return expect

        """maximizing task for the root node"""
        max = float("-inf")
        action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            maxV = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if maxV > max or max == float("-inf"):
                max = maxV
                action = agentState

        return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Distance to Food
    newfoodList = util.matrixAsList(newFood.data) # Locations of all food
    min_food_distance = 1 
    closeFoodDist = 999999
    farFoodDist = 1
    for food in newfoodList:
        dist = util.manhattanDistance(newPos, food)
        min_food_distance += dist
        if dist < closeFoodDist:
            closeFoodDist = dist
        if dist > farFoodDist:
            farFoodDist = dist

    # Distance to ghosts
    GhostDist = 1 # init to 1 to prevent div by 0
    CloseGhost = 0
    scaredGhost = 10
    for ghostIdx in range(len(newGhostStates)):
        ghostState = newGhostStates[ghostIdx]
        scaredTime = newScaredTimes[ghostIdx]
        if scaredTime > 0: # If ghost in deactive state, don't care about it's distance
            scaredGhost += 50
            continue
        else:
            ghostDist = util.manhattanDistance(newPos, ghostState.configuration.pos)
            GhostDist += ghostDist
            if ghostDist <= 1:
                CloseGhost -= 200 # negative number
    
    return currentGameState.getScore() + (1/float(closeFoodDist)) - 0.3*(1/float(GhostDist)) + CloseGhost + scaredGhost


    
# Abbreviation
better = betterEvaluationFunction