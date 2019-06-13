# Look for #IMPLEMENT tags in this file. These tags indicate what has
# to be implemented to complete the LunarLockout  domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

# import os for time functions
from search import *  # for search engines
from lunarlockout import LunarLockoutState, Direction, \
    lockout_goal_state  # for LunarLockout specific classes and problems


# LunarLockout HEURISTICS
def heur_trivial(state: LunarLockoutState):
    '''trivial admissible LunarLockout heuristic'''
    '''INPUT: a LunarLockout state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    return 0


def heur_manhattan_distance(state: LunarLockoutState):
    # OPTIONAL
    '''Manhattan distance LunarLockout heuristic'''
    '''INPUT: a lunar lockout state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # Write a heuristic function that uses Manhattan distance to estimate distance between the current state and the goal.
    # Your function should return a sum of the Manhattan distances between each xanadu and the escape hatch.

    distance = 0
    center = int((state.width - 1) / 2)
    for rover in state.xanadus:
        distance += abs(center - rover[0]) + abs(center - rover[1])

    return distance


def heur_L_distance(state: LunarLockoutState):
    # IMPLEMENT
    '''L distance LunarLockout heuristic'''
    '''INPUT: a lunar lockout state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # Write a heuristic function that uses mahnattan distance to estimate distance between the current state and the goal.
    # Your function should return a sum of the L distances between each xanadu and the escape hatch.
    distance = 0
    center = int((state.width - 1) / 2)
    for rover in state.xanadus:
        if rover[0] != center:
            distance += 1
        if rover[1] != center:
            distance += 1
    return distance


def heur_alternate(state: LunarLockoutState):
    # IMPLEMENT
    '''a better lunar lockout heuristic'''
    '''INPUT: a lunar lockout state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.
    
    if a rover is not in the same row as the center:
        then it takes at least one move to move to the center row
        h += 1
        if on the center row, no other robot could stop the rover at the center:
            then it takes at least one move to move some robot to that position
            h += 1 
            
    Note: if the rover is in the same row or same column as the escape hatch, then no h-val is added for that direction
    the above is one of the two case (try to move vertically then horizontally to reach the center)
    we need to consider the other case (try to move horizontally then vertically to reach the center)
    the algorithm is exactly the same (just change row to column)
    
    in addition, to avoid repeat-counting, we have a memoization array M, every position checked (to see if a helper 
    robot exists) won't be checked again and h-value won't be counted again for the same position.
    
    unsolvable case is checked, if a rover is at a position that's furtherest from the escape hatch than any other robots (no one could help
    it to get closer to center), then return infinity for the unsolvable state.

    Since every h-val increased is minimized (at least...), the h-value must be admissible
    '''

    distance = 0
    center = int((state.width - 1) / 2)
    M = []
    map = []
    for i in range(state.width):
        M.append([])
        map.append([])
        for j in range(state.height):
            M[i].append(False)
            map[i].append(False)

    for rover in state.xanadus:
        map[rover[0]][rover[1]] = True
    for robot in state.robots:
        map[robot[0]][robot[1]] = True

    if unsolvable(state):
        return float("inf")

    for rover in state.xanadus:
        distance += heur_alternate_helper(M, map, center, rover)

    return distance


def unsolvable(state: LunarLockoutState):
    ''' check if a given state is unsolvable
        if a rover is at a position where no other robots/rover could help it to move closer to the escape hatch, then it's unsolvable
    '''
    for rover in state.xanadus:
        if gt_all(rover, state, 0):
            if gt_all(rover, state, 1) or lt_all(rover, state, 1):
                return True

        if lt_all(rover, state, 0):
            if gt_all(rover, state, 1) or lt_all(rover, state, 1):
                return True
        
        return False


def gt_all(rover, state: LunarLockoutState, x_or_y: int):
    ''' check if the position index of a given rover is greater than all other robots/rovers in a given direction (row or column)
    '''
    for robot in state.xanadus:
        if rover[x_or_y] < robot[x_or_y]:
            return False
    for robot in state.robots:
        if rover[x_or_y] < robot[x_or_y]:
            return False
    return True


def lt_all(rover, state: LunarLockoutState, x_or_y: int):
    ''' check if the position index of a given rover is less than all other robots/rovers in a given direction (row or column)'''
    for robot in state.xanadus:
        if rover[x_or_y] > robot[x_or_y]:
            return False
    for robot in state.robots:
        if rover[x_or_y] > robot[x_or_y]:
            return False
    return True


def heur_alternate_helper(M: list, map: list, center: int, rover: tuple) -> int:
    '''determine heuristic cost to be added for a rover based on positions of other robots and rovers'''
    distance = [0, 0]
    distance[0] += estimate_heur_horizontal_distance(M, map, center, rover)
    if distance[0] == 0:
        distance[0] += estimate_heur_vertical_distance(M, map, center, rover)
    else:
        new_pos = (center, rover[1])
        distance[0] += estimate_heur_vertical_distance(M, map, center, new_pos)

    distance[1] += estimate_heur_vertical_distance(M, map, center, rover)
    if distance[1] == 0:
        distance[1] += estimate_heur_horizontal_distance(M, map, center, rover)
    else:
        new_pos = (rover[0], center)
        distance[1] += estimate_heur_horizontal_distance(M, map, center, new_pos)

    return min(distance[0], distance[1])


def estimate_heur_horizontal_distance(M: list, map, center: int, pos: tuple) -> int:
    '''given a rover's position, if there is a robot/rover on the save row that could stop the rover on the center column, then
        return 0, otherwise return 1
    '''
    distance = 1
    if center == pos[0]:
        return 0
    else:
        if center > pos[0]:
            horizontal_pos = center + 1
        else:
            horizontal_pos = center - 1
        if M[horizontal_pos][pos[1]]:
            return distance
        if not map[horizontal_pos][pos[1]]:
            distance += 1
        M[horizontal_pos][pos[1]] = True
    return distance


def estimate_heur_vertical_distance(M: list, map, center: int, pos: tuple) -> int:
    '''given a rover's position, if there is a robot/rover on the same column that could stop the rover on the center row, then
        return 0, otherwise return 1
    '''
    distance = 1
    if center == pos[1]:
        return 0
    else:
        if center > pos[1]:
            vertical_pos = center + 1
        else:
            vertical_pos = center - 1
        if M[pos[0]][vertical_pos]:
            return distance
        if not map[pos[0]][vertical_pos]:
            distance += 1
        M[pos[0]][vertical_pos] = True
    return distance


def fval_function(sN: sNode, weight: float):
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a LunarLockoutState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """

    # Many searches will explore nodes (or states) that are ordered by their f-value.
    # For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue is the hval of the state.
    # You can use this function to create an alternate f-value for states; this must be a function of the state and the weight.
    # The function must return a numeric f-value.
    # The value will determine your state's position on the Frontier list during a 'custom' search.
    # You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.
    return sN.gval + weight * sN.hval


def anytime_weighted_astar(initial_state, heur_fn, weight=4., timebound=2):
    # IMPLEMENT
    '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
    '''INPUT: a lunar lockout state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of weighted astar algorithm'''
    se = SearchEngine('custom')
    wrapped_fval_function = lambda sN: fval_function(sN, weight)
    # se.init_search(initial_state, lockout_goal_state, heur_fn, wrapped_fval_function)
    best_so_far = None
    search_stop_time = os.times()[0] + timebound
    while search_stop_time > os.times()[0] and weight >= 1:
        se.init_search(initial_state, lockout_goal_state, heur_fn, wrapped_fval_function)
        result = se.search(search_stop_time - os.times()[0])
        if result:
            if best_so_far is None:
                best_so_far = result
            if best_so_far.gval > result.gval:
                best_so_far = result
        weight = 1 + (weight - 1) * 4 / 5
    return best_so_far


def anytime_gbfs(initial_state, heur_fn, timebound=2):
    # OPTIONAL
    '''Provides an implementation of anytime greedy best-first search.  This iteratively uses greedy best first search,'''
    '''At each iteration, however, a cost bound is enforced.  At each iteration the cost of the current "best" solution'''
    '''is used to set the cost bound for the next iteration.  Only paths within the cost bound are considered at each iteration.'''
    '''INPUT: a lunar lockout state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    return 0


PROBLEMS = (
    # 5x5 boards: all are solveable
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((0, 1),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((0, 2),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((0, 3),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((1, 1),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((1, 2),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((1, 3),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((1, 4),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((2, 0),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((2, 1),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (0, 2), (0, 4), (2, 0), (4, 0)), ((4, 4),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((4, 0),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((4, 1),)),
    LunarLockoutState("START", 0, None, 5, ((0, 0), (1, 0), (2, 2), (4, 2), (0, 4), (4, 4)), ((4, 3),)),
    # 7x7 BOARDS: all are solveable
    LunarLockoutState("START", 0, None, 7, ((4, 2), (1, 3), (6, 3), (5, 4)), ((6, 2),)),
    LunarLockoutState("START", 0, None, 7, ((2, 1), (4, 2), (2, 6)), ((4, 6),)),
    LunarLockoutState("START", 0, None, 7, ((2, 1), (3, 1), (4, 1), (2, 6), (4, 6)), ((2, 0), (3, 0), (4, 0))),
    LunarLockoutState("START", 0, None, 7, ((1, 2), (0, 2), (2, 3), (4, 4), (2, 5)), ((2, 4), (3, 1), (4, 0))),
    LunarLockoutState("START", 0, None, 7, ((3, 2), (0, 2), (3, 3), (4, 4), (2, 5)), ((1, 2), (3, 0), (4, 0))),
    LunarLockoutState("START", 0, None, 7, ((3, 1), (0, 2), (3, 3), (4, 4), (2, 5)), ((1, 2), (3, 0), (4, 0))),
    LunarLockoutState("START", 0, None, 7, ((2, 1), (0, 2), (1, 2), (6, 4), (2, 5)), ((2, 0), (3, 0), (4, 0))),
)

if __name__ == "__main__":

    # TEST CODE
    solved = 0;
    unsolved = [];
    counter = 0;
    percent = 0;
    timebound = 2;  # 2 second time limit for each problem
    print("*************************************")
    print("Running A-star")

    for i in range(len(
            PROBLEMS)):  # note that there are 40 problems in the set that has been provided.  We just run through 10 here for illustration.

        print("*************************************")
        print("PROBLEM {}".format(i))

        s0 = PROBLEMS[i]  # Problems will get harder as i gets bigger

        print("*******RUNNING A STAR*******")
        se = SearchEngine('astar', 'full')
        se.init_search(s0, lockout_goal_state, heur_alternate)
        final = se.search(timebound)

        if final:
            final.print_path()
            solved += 1
        else:
            unsolved.append(i)
        counter += 1

    if counter > 0:
        percent = (solved / counter) * 100

    print("*************************************")
    print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))
    print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))
    print("*************************************")

    solved = 0;
    unsolved = [];
    counter = 0;
    percent = 0;
    print("Running Anytime Weighted A-star")

    for i in range(len(PROBLEMS)):
        print("*************************************")
        print("PROBLEM {}".format(i))

        s0 = PROBLEMS[i]
        weight = 4
        final = anytime_weighted_astar(s0, heur_alternate, weight, timebound)

        if final:
            final.print_path()
            solved += 1
        else:
            unsolved.append(i)
        counter += 1

    if counter > 0:
        percent = (solved / counter) * 100

    print("*************************************")
    print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))
    print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))
    print("*************************************")

    solved = 0;
    unsolved = [];
    counter = 0;
    percent = 0;
    print("Running Anytime GBFS")

    for i in range(len(PROBLEMS)):
        print("*************************************")
        print("PROBLEM {}".format(i))

        s0 = PROBLEMS[i]
        final = anytime_gbfs(s0, heur_alternate, timebound)

        if final:
            final.print_path()
            solved += 1
        else:
            unsolved.append(i)
        counter += 1

    if counter > 0:
        percent = (solved / counter) * 100

    print("*************************************")
    print("{} of {} problems ({} %) solved in less than {} seconds.".format(solved, counter, percent, timebound))
    print("Problems that remain unsolved in the set are Problems: {}".format(unsolved))
    print("*************************************")
