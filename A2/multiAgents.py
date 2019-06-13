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


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    direction_dict = {"Stop": (0, 0), "North": (0, 1), "South": (0, -1), "West": (-1, 0), "East": (1, 0)}
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        # initialize some useful value
        print("==================================================")
        current_pos = currentGameState.getPacmanPosition()
        print("current_pos=", current_pos)
        print("newPos=", newPos)
        current_food_list = currentGameState.getFood().asList()
        # print("current_food_list=", current_food_list)
        new_food_list = newFood.asList()
        # print("new_food_list=", new_food_list)
        width = newFood.width
        print("width=", width)
        height = newFood.height
        print("height=", height)
        direction = self.direction_dict[action]
        print("direction=", action, direction)
        current_ghost_positions = currentGameState.getGhostPositions()
        new_ghost_positions = successorGameState.getGhostPositions()
        danger_zone_M_distance = max(1, width * height / 50)
        print("danger_zone_M_distance=", danger_zone_M_distance)
        closest_ghost_m_distance = self.closestGhostMDistance(new_ghost_positions, newPos)
        # print("==================================================")
        score = 0
        # check scared time
        total_scared_time = 0
        for scared_time in newScaredTimes:
            total_scared_time += scared_time
        score += total_scared_time
        if closest_ghost_m_distance <= 1 and total_scared_time == 0:
            return 0
        # see if newPos has food in currentState's view
        food_in_new_pos_score = 0
        if currentGameState.hasFood(newPos[0], newPos[1]):
            print("hasFood: True")
            food_in_new_pos_score += 100
            score += food_in_new_pos_score
        else:
            print("hasFood: False")
        if len(new_ghost_positions) == 0:
            sum_food_in_range = self.sumOfFoodInActionDirectionRange(action, currentGameState, current_food_list,
                                                                     newFood,
                                                                     current_pos, direction)
            score += sum_food_in_range / danger_zone_M_distance
            return score + random.randint(0, 10)
        # find closest food with manhattan distance
        new_pos_closest_food_m_distance = self.closestFoodMDistance(newPos, new_food_list)
        score += 1 / new_pos_closest_food_m_distance * 100  # reciprocal of distance, lower distance => higher score
        print("score after closest food distance=", score)

        # if in a zone, no food around, check greater range for food
        if score <= danger_zone_M_distance:
            # food in range of direction of action
            sum_food_in_range = self.sumOfFoodInActionDirectionRange(action, currentGameState, current_food_list,
                                                                     newFood,
                                                                     current_pos, direction)
            score += sum_food_in_range
            score += random.randint(0, 5)

        count_wall = 0
        if successorGameState.hasWall(newPos[0], newPos[1] + 1):
            count_wall += 1
        if successorGameState.hasWall(newPos[0], newPos[1] - 1):
            count_wall += 1
        if successorGameState.hasWall(newPos[0] + 1, newPos[1]):
            count_wall += 1
        if successorGameState.hasWall(newPos[0] - 1, newPos[1]):
            count_wall += 1
        print("num wall around: ", count_wall)

        if count_wall >= 3:
            return food_in_new_pos_score
        # to void ghost

        if closest_ghost_m_distance <= danger_zone_M_distance and total_scared_time == 0:
            score = min(score, closest_ghost_m_distance)
        print("Score: ", score)
        print("==================================================")
        return score

    def closestFoodMDistance(self, newPos: tuple, new_food_list: list) -> int:
        min_distance = float("inf")
        for pos in new_food_list:
            tmp_distance = manhattanDistance(newPos, pos)
            min_distance = min(min_distance, tmp_distance)
        return min_distance

    def closestGhostMDistance(self, new_ghost_positions, pacman_pos) -> int:
        min_distance = float("inf")
        for pos in new_ghost_positions:
            tmp_distance = manhattanDistance(pos, pacman_pos)
            min_distance = min(tmp_distance, min_distance)
        return min_distance

    def foodAround(self, new_food_list, newPos, successorGameState):
        count_food_around = 0
        offset_lst = [-2, -1, 0, 1, 2]
        for offset_x in offset_lst:
            for offset_y in offset_lst:
                if successorGameState.hasFood(offset_x + newPos[0], offset_y + newPos[1]):
                    count_food_around += 1
        return count_food_around

    def sumOfFoodInActionDirectionRange(self, action, currentGameState, current_food_list, newFood, current_pos,
                                        direction):
        print("current_pos: " + str(current_pos))
        search_range_x = (0, newFood.width)
        search_range_y = (0, newFood.height)
        if direction[0] == 1:
            search_range_x = (currentGameState.getPacmanPosition()[0] + 1, newFood.width)
        elif direction[0] == -1:
            search_range_x = (1, currentGameState.getPacmanPosition()[0] - 1)
        elif direction[1] == 1:
            search_range_y = (currentGameState.getPacmanPosition()[1] + 1, newFood.height)
        elif direction[1] == -1:
            search_range_y = (1, currentGameState.getPacmanPosition()[1] - 1)
        print("width: ", newFood.width)
        print("height: ", newFood.height)
        print("x_range: " + str(search_range_x))
        print("y_range: " + str(search_range_y))
        # print("current_food_list: ", current_food_list)
        sum_food_in_range = 0
        if action is not Directions.STOP:
            sum_food_in_range = self.sumFoodGivenRange(search_range_x, search_range_y, current_food_list)
        print("sum_food_in_range: " + str(sum_food_in_range))
        return sum_food_in_range

    def sumFoodGivenRange(self, search_range_x, search_range_y, current_food_list):
        count_food = 0
        for food in current_food_list:
            if search_range_x[1] >= food[0] >= search_range_x[0] and search_range_y[1] >= food[1] >= search_range_y[0]:
                count_food += 1
        return count_food


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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
        num_ghost = gameState.getNumAgents() - 1
        score, action = self.DFMiniMax(gameState, "pacman", num_ghost, 0, 0)
        return action

    def DFMiniMax(self, curr_game_state, player: str, num_ghost: int, ghost_index: int, depth_so_far: int):
        # print("depth_so_far: ", depth_so_far)
        if curr_game_state.isLose() or curr_game_state.isWin() or depth_so_far > self.depth:
            return self.evaluationFunction(curr_game_state), None
        # generate new states using legal moves
        if player == "pacman":
            # print("pacman at depth ", depth_so_far)
            pacman_legal_moves = curr_game_state.getLegalActions(0)  # pacman legal moves
            # pacman_successors = []  # list of game state after pacman's action
            max_score = -float("inf")
            best_action = None
            for action in pacman_legal_moves:
                successor_game_state = curr_game_state.generateSuccessor(0, action)
                tmp_score, next_action = self.DFMiniMax(successor_game_state, "ghost", num_ghost, ghost_index + 1,
                                                        depth_so_far + 1)
                if tmp_score > max_score:
                    max_score, best_action = tmp_score, action
            return max_score, best_action



        else:  # is ghost
            # print("ghost ", ghost_index, " at depth ", depth_so_far)
            ghost_legal_moves = curr_game_state.getLegalActions(ghost_index)
            # ghost_successors = []  # list of game state after ghost's action
            min_score = float("inf")
            best_action = None
            for action in ghost_legal_moves:
                successor_game_state = curr_game_state.generateSuccessor(ghost_index, action)
                if ghost_index == num_ghost:

                    if depth_so_far == self.depth:
                        depth_so_far += 1
                    tmp_score, _ = self.DFMiniMax(successor_game_state, "pacman", num_ghost, 0,
                                                  depth_so_far)  # restart a new depth
                else:
                    tmp_score, _ = self.DFMiniMax(successor_game_state, "ghost", num_ghost, ghost_index + 1,
                                                  depth_so_far)
                if tmp_score < min_score:
                    min_score, best_action = tmp_score, action
            return min_score, best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


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
