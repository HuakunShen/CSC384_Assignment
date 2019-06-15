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

        # Print out these variables to see what you're getting, then combine them
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
        # print("==================================================")
        current_pos = currentGameState.getPacmanPosition()
        # print("current_pos=", current_pos)
        # print("newPos=", newPos)
        current_food_list = currentGameState.getFood().asList()
        #         # print("current_food_list=", current_food_list)
        new_food_list = newFood.asList()
        #         # print("new_food_list=", new_food_list)
        width = newFood.width
        # print("width=", width)
        height = newFood.height
        # print("height=", height)
        direction = self.direction_dict[action]
        # print("direction=", action, direction)
        current_ghost_positions = currentGameState.getGhostPositions()
        new_ghost_positions = successorGameState.getGhostPositions()
        danger_zone_M_distance = max(1, width * height / 50)
        # print("danger_zone_M_distance=", danger_zone_M_distance)
        closest_ghost_m_distance = self.closestGhostMDistance(new_ghost_positions, newPos)
        #         # print("==================================================")
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
            # print("hasFood: True")
            food_in_new_pos_score += 100
            score += food_in_new_pos_score
        # else:
        # print("hasFood: False")
        if len(new_ghost_positions) == 0:
            sum_food_in_range = self.sumOfFoodInActionDirectionRange(action, currentGameState, current_food_list,
                                                                     newFood,
                                                                     current_pos, direction)
            score += sum_food_in_range / danger_zone_M_distance
            return score + random.randint(0, 10)
        # find closest food with manhattan distance
        new_pos_closest_food_m_distance = self.closestFoodMDistance(newPos, new_food_list)
        score += 1 / new_pos_closest_food_m_distance * 100  # reciprocal of distance, lower distance => higher score
        # print("score after closest food distance=", score)

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
        # print("num wall around: ", count_wall)

        if count_wall >= 3:
            return food_in_new_pos_score
        # to void ghost

        if closest_ghost_m_distance <= danger_zone_M_distance and total_scared_time == 0:
            score = min(score, closest_ghost_m_distance)
        # print("Score: ", score)
        # print("==================================================")
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

    def sumOfFoodInActionDirectionRange(self, action, currentGameState, current_food_list, newFood, current_pos,
                                        direction):
        # print("current_pos: " + str(current_pos))
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
        # print("width: ", newFood.width)
        # print("height: ", newFood.height)
        # print("x_range: " + str(search_range_x))
        # print("y_range: " + str(search_range_y))
        # # print("current_food_list: ", current_food_list)
        sum_food_in_range = 0
        if action is not Directions.STOP:
            sum_food_in_range = self.sumFoodGivenRange(search_range_x, search_range_y, current_food_list)
        # print("sum_food_in_range: " + str(sum_food_in_range))
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
        legal_moves = gameState.getLegalActions(0)
        best_action = legal_moves[0]
        _, tmp_action = self.pacmanMiniMax(gameState, num_ghost, 0)
        best_action = tmp_action if tmp_action is not None else best_action
        return best_action

    def DFMiniMax(self, game_state, agent_index: int, num_ghost: int, depth_so_far: int) -> int:
        ''' given a state and a player, return the max score it can get '''
        if depth_so_far > self.depth or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state)

        if agent_index == 0:  # pacman
            return self.pacmanMiniMax(game_state, num_ghost, depth_so_far)[0]
        else:  # ghost
            return self.ghostMiniMax(game_state, agent_index, num_ghost, depth_so_far)

    def pacmanMiniMax(self, game_state, num_ghost: int, depth_so_far: int):
        best_action = None
        legal_moves = game_state.getLegalActions(0)
        max_score = -float("inf")
        next_agent = 1 if num_ghost != 0 else 0  # in case there is no ghost
        for action in legal_moves:
            successor_state = game_state.generateSuccessor(0, action)
            tmp_score = self.DFMiniMax(successor_state, next_agent, num_ghost, depth_so_far + 1)
            if tmp_score > max_score:
                max_score, best_action = tmp_score, action
        return max_score, best_action

    def ghostMiniMax(self, game_state, agent_index: int, num_ghost: int, depth_so_far: int):
        legal_moves = game_state.getLegalActions(agent_index)
        min_score = float("inf")
        next_agent = agent_index + 1 if agent_index != num_ghost else 0  # in case there is no ghost
        # if it's the last ghost on the last depth, +1 to depth_so_far to quit early in next DFMiniMax
        next_depth = depth_so_far + 1 if agent_index == num_ghost and depth_so_far == self.depth else depth_so_far
        for action in legal_moves:
            successor_state = game_state.generateSuccessor(agent_index, action)
            tmp_score = self.DFMiniMax(successor_state, next_agent, num_ghost, next_depth)
            if tmp_score < min_score:
                min_score = tmp_score
        return min_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_ghost = gameState.getNumAgents() - 1
        legal_moves = gameState.getLegalActions(0)
        best_action = legal_moves[0]
        alpha = -float("inf")
        beta = float("inf")
        _, tmp_action = self.pacmanAlphaBeta(gameState, num_ghost, 0, alpha, beta)
        best_action = tmp_action if tmp_action is not None else best_action
        # if best_action == Directions.STOP:
        # print("STOP")
        return best_action

    def AlphaBetaPruning(self, game_state, agent_index: int, num_ghost: int, depth_so_far: int, alpha: float,
                         beta: float) -> float:
        # if terminal state or depth reached limit, return current score
        if depth_so_far > self.depth or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state)
        if agent_index == 0:  # pacman
            return self.pacmanAlphaBeta(game_state, num_ghost, depth_so_far, alpha, beta)[0]
        else:  # ghost
            return self.ghostAlphaBeta(game_state, agent_index, num_ghost, depth_so_far, alpha, beta)

    def pacmanAlphaBeta(self, game_state, num_ghost: int, depth_so_far: int, alpha: float, beta: float):
        # pacman corresponds to Max/alpha, it can only modify alpha's value
        best_action = None
        legal_moves = game_state.getLegalActions(0)
        next_agent = 1 if num_ghost != 0 else 0  # in case there is no ghost
        for action in legal_moves:
            successor_state = game_state.generateSuccessor(0, action)
            tmp_score = self.AlphaBetaPruning(successor_state, next_agent, num_ghost, depth_so_far + 1, alpha, beta)
            if tmp_score > alpha:
                alpha, best_action = tmp_score, action
            if beta <= alpha:
                break
        return alpha, best_action

    def ghostAlphaBeta(self, game_state, agent_index, num_ghost: int, depth_so_far: int, alpha: float,
                       beta: float) -> float:
        # ghost corresponds to min/beta, it can only modify beta's value
        legal_moves = game_state.getLegalActions(agent_index)
        next_agent = agent_index + 1 if agent_index != num_ghost else 0  # in case there is no ghost
        # if it's the last ghost on the last depth, +1 to depth_so_far to quit early in next DFMiniMax
        next_depth = depth_so_far + 1 if agent_index == num_ghost and depth_so_far == self.depth else depth_so_far
        for action in legal_moves:
            successor_state = game_state.generateSuccessor(agent_index, action)
            tmp_score = self.AlphaBetaPruning(successor_state, next_agent, num_ghost, next_depth, alpha, beta)
            if tmp_score < beta:
                beta = tmp_score
            if beta <= alpha:
                break
        return beta


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
        num_ghost = gameState.getNumAgents() - 1
        legal_moves = gameState.getLegalActions(0)
        best_action = legal_moves[random.randint(0, len(legal_moves) - 1)]
        _, tmp_action = self.expectimax(gameState, 0, num_ghost, 0)
        best_action = tmp_action if tmp_action is not None else best_action
        # print("action: ", best_action)
        return best_action

    def expectimax(self, game_state, agent_index: int, num_ghost: int, depth_so_far: int):
        best_move = None
        if depth_so_far > self.depth or game_state.isLose() or game_state.isWin():
            return self.evaluationFunction(game_state), best_move
        value = -float("inf") if agent_index == 0 else float(0)
        legal_moves = game_state.getLegalActions(agent_index)
        next_agent = agent_index + 1 if agent_index != num_ghost else 0
        # next_depth = depth_so_far
        if agent_index == 0 or (agent_index == num_ghost and depth_so_far == self.depth):
            depth_so_far += 1
        probability_per_ghost = 1.0 / float(len(legal_moves))
        #         # print("prob: ", probability_per_ghost)

        all_best_moves = []
        for action in legal_moves:
            successor_state = game_state.generateSuccessor(agent_index, action)
            nxt_val, nxt_move = self.expectimax(successor_state, next_agent, num_ghost, depth_so_far)
            if agent_index == 0 and value <= nxt_val:
                if value < nxt_val:
                    all_best_moves = [action]
                else:
                    all_best_moves.append(action)
                # value, best_move = nxt_val, action
                value = nxt_val
            if agent_index != 0:
                value += float(probability_per_ghost) * float(nxt_val)
        if all_best_moves:
            best_move = all_best_moves[random.randint(0, len(all_best_moves) - 1)]
        return value, best_move


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      factors to consider:
      1. adjacent food
      2. closest food (Manhattan distance)
      3. adjacent ghost => escape immediately
      4. closest ghost distance
      5. number of ghost within a specific range (range scale to the size of map)
      6. number of food around the pacman (range scale to the size of map)
      7. scare time (if > 0 => most ghost are scared)
    """
    "*** YOUR CODE HERE ***"
    score = 0
    if currentGameState.isLose():
        score = -float("inf")
        return score

    if currentGameState.isWin():
        score = float("inf")
        return score
    # print("===============================================")
    # initialization of some useful data
    current_food = currentGameState.getFood()
    food_list = current_food.asList()
    width = current_food.width
    height = current_food.height
    pacman_position = currentGameState.getPacmanPosition()
    # print("pacman position: ", pacman_position)
    radius = int(0.2619 * (float(width) * float(height)) ** 0.3777)
    ghost_positions = currentGameState.getGhostPositions()
    # print("ghost positions: ", ghost_positions)
    ghost_states = currentGameState.getGhostStates()
    scared_times_list = [ghostState.scaredTimer for ghostState in ghost_states]

    # scare time for ghost hunting
    total_scared_time = 0
    all_scared = True
    for scared_time in scared_times_list:
        if scared_time == 0:
            all_scared = False
        total_scared_time += scared_time
    # print("total scared time: ", total_scared_time, "=== all scared: ", all_scared)

    # number of food left on map
    num_food_left = current_food.count()
    # print("num_food_left: ", num_food_left)
    # score += width * height - num_food_left + random.randint(0, 1)
    # important to times 10 in the next line, scale up the importance of food left. Without it, AI wins much slower
    score += (width * height - num_food_left) * 5

    # check closest food, closer is better, under the condition that there is not food around
    num_food_around = foodAround(pacman_position, currentGameState, 1)
    # print("num_food_around: ", num_food_around)
    # priority given to eating foods around, if there is no food around, run towards the closest food
    if num_food_around == 0:
        closest_food_distance = closestFoodMDistance(pacman_position, food_list)
        score -= closest_food_distance / 2  # closer is better






    # if all_scared:  # want to eat the pill that scares ghost
    #     score *= 2
    #     ghost_is_in_view, ghost_in_view_pos = ghostInView(ghost_positions, pacman_position, currentGameState)
    #     if ghost_is_in_view:
    #         score *= 2
    # elif total_scared_time == 0:  # not all scared
    #     closest_ghost_distance = closestGhostMDistance(ghost_positions, pacman_position)
    #     if closest_ghost_distance <= 1:
    #         score = 0
    closest_ghost_distance = closestGhostMDistance(ghost_positions, pacman_position)
    if closest_ghost_distance < max(width, height) / 2 and total_scared_time > 0:
        score *= 2

    if not all_scared and closest_ghost_distance <= 1:
        score = 0






    # check num of walls around
    # wall_adjacent = num_wall_adjacent(currentGameState, pacman_position)
    # print("num wall around: ", wall_adjacent)
    # if wall_adjacent == 3:
    #     score /= 2

    # print("score: ", score)
    return score + currentGameState.getScore()


direction_dict = {Directions.STOP: (0, 0), Directions.NORTH: (0, 1), Directions.SOUTH: (0, -1),
                  Directions.WEST: (-1, 0), Directions.EAST: (1, 0)}


def ghostInView(ghost_positions, pacman_position, game_state):
    cur_pos = pacman_position
    for ghost_pos in ghost_positions:
        if cur_pos[0] == ghost_pos[0]:  # in the same column
            y_offset = 1 if cur_pos[1] < ghost_pos[1] else -1
            while not game_state.hasWall(cur_pos[0], cur_pos[1]):
                if cur_pos == ghost_pos:
                    return True, ghost_pos
                cur_pos = (cur_pos[0], cur_pos[1] + y_offset)
        elif cur_pos[1] == ghost_pos[1]:  # in the same row
            x_offset = 1 if cur_pos[0] < ghost_pos[0] else -1
            while not game_state.hasWall(cur_pos[0], cur_pos[1]):
                if cur_pos == ghost_pos:
                    return True, ghost_pos
                cur_pos = (cur_pos[0] + x_offset, cur_pos[1])
    return False, None


def num_wall_adjacent(game_state, position):
    count_wall = 0
    if game_state.hasWall(position[0], position[1] + 1):
        count_wall += 1
    if game_state.hasWall(position[0], position[1] - 1):
        count_wall += 1
    if game_state.hasWall(position[0] + 1, position[1]):
        count_wall += 1
    if game_state.hasWall(position[0] - 1, position[1]):
        count_wall += 1
    return count_wall


def closest_food_distance_straight_line(currentGameState, pacman_pos) -> int:
    legal_moves = currentGameState.getLegalActions()
    min_distance = float("inf")
    for action in legal_moves:
        if action == Directions.STOP:
            continue
        direction = direction_dict[action]
        # print("direction: ", direction)
        cur_pos = pacman_pos
        distance = 0
        while not currentGameState.hasWall(cur_pos[0], cur_pos[1]):
            # print("position checked: ", cur_pos)
            # print("direction: ", direction)
            if currentGameState.hasFood(cur_pos[0], cur_pos[1]):
                min_distance = min(distance, min_distance)
            cur_pos = (cur_pos[0] + direction[0], cur_pos[1] + direction[1])
            distance += 1
    return min_distance


def num_ghost_around(ghost_positions, pacman_position, radius: int):
    num_ghost = 0
    x_range = (pacman_position[0] - radius, pacman_position[0] + radius)
    y_range = (pacman_position[1] - radius, pacman_position[1] + radius)
    for ghost in ghost_positions:
        if x_range[0] <= ghost[0] <= x_range[1] and y_range[0] <= ghost[1] <= y_range[1]:
            num_ghost += 1
    return num_ghost


def foodAround(position, game_state, radius: int):
    count_food_around = 0
    offset_lst = []
    # initialize offset_lst based on radius
    for i in range(radius, 0, -1):
        offset_lst.append(-i)
    for i in range(0, radius + 1):
        offset_lst.append(i)
    # print("offset list: ", offset_lst)
    for offset_x in offset_lst:
        for offset_y in offset_lst:
            # print("position checked: ", (offset_x + position[0], offset_y + position[1]))
            if game_state.hasFood(offset_x + position[0], offset_y + position[1]):
                count_food_around += 1
    return count_food_around


def closestFoodMDistance(position: tuple, food_list: list) -> int:
    min_distance = float("inf")
    for pos in food_list:
        tmp_distance = manhattanDistance(position, pos)
        min_distance = min(min_distance, tmp_distance)
    return min_distance


def closestGhostMDistance(ghost_positions, pacman_pos) -> int:
    min_distance = float("inf")
    for pos in ghost_positions:
        tmp_distance = manhattanDistance(pos, pacman_pos)
        min_distance = min(tmp_distance, min_distance)
    return min_distance


# Abbreviation
better = betterEvaluationFunction
