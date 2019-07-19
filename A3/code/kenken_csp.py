# Look for #IMPLEMENT tags in this file.
'''
All models need to return a CSP object, and a list of lists of Variable objects 
representing the board. The returned list of lists is used to access the 
solution. 

For example, after these three lines of code

    csp, var_array = kenken_csp_model(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the KenKen puzzle.

The grid-only models do not need to encode the cage constraints.

1. binary_ne_grid (worth 10/100 marks)
    - A model of a KenKen grid (without cage constraints) built using only 
      binary not-equal constraints for both the row and column constraints.

2. nary_ad_grid (worth 10/100 marks)
    - A model of a KenKen grid (without cage constraints) built using only n-ary 
      all-different constraints for both the row and column constraints. 

3. kenken_csp_model (worth 20/100 marks) 
    - A model built using your choice of (1) binary binary not-equal, or (2) 
      n-ary all-different constraints for the grid.
    - Together with KenKen cage constraints.

'''
from cspbase import *
import itertools


def binary_ne_grid(kenken_grid):
    board_dim = kenken_grid[0][0]
    board_list = []
    domain = list(range(1, board_dim + 1))  # construct domain for every variable
    all_vars = []
    for row_i in range(board_dim):
        row = []
        for col in range(board_dim):
            var = Variable(str(row_i + 1) + str(col + 1), domain[:])
            row.append(var)
            all_vars.append(var)  # for csp construction
        board_list.append(row)

    # construct csp object
    csp = CSP('binary_ne', all_vars)
    # constraints
    # rows
    all_satisfied = []
    tmp_iter = [domain, domain]
    for t in itertools.product(*tmp_iter):  # construct a list of possible satisfied values
        if t[0] != t[1]:
            all_satisfied.append(t)

    for row in board_list:
        for i in range(board_dim):
            first = row[i]
            for j in range(board_dim):
                if i != j:
                    second = row[j]
                    variables = [first, second]
                    constraint = Constraint('C_Row' + str(i) + "_" + str(i) + str(j), variables)
                    constraint.add_satisfying_tuples(all_satisfied)
                    csp.add_constraint(constraint)

    # cols
    for col_index in range(board_dim):
        col = []  # every col_index needs only one col list
        for row in board_list:  # for each row, take the element at col_index and put into a col list
            col.append(row[col_index])
        for i in range(len(col)):
            first = col[i]
            for j in range(len(col)):
                if i != j:
                    second = col[j]
                    variables = [first, second]
                    constraint = Constraint('C_Col' + str(i) + "_" + str(i) + str(j), variables)
                    constraint.add_satisfying_tuples(all_satisfied)
                    csp.add_constraint(constraint)
    return csp, board_list


def nary_ad_grid(kenken_grid):
    board_dim = kenken_grid[0][0]
    board_list = []
    domain = list(range(1, board_dim + 1))  # construct domain for every variable, is every number initially
    all_vars = []
    # put every vars into board_list, board_list is 2-dimensional
    for row_i in range(board_dim):
        row = []
        for col in range(board_dim):
            var = Variable(str(row_i + 1) + str(col + 1), domain[:])
            row.append(var)
            all_vars.append(var)  # for csp construction
        board_list.append(row)

    # construct csp object containing every variable created in board_list
    csp = CSP('nary_ad', all_vars)
    all_satisfied = list(itertools.permutations(domain))
    # row
    for i in range(board_dim):
        row = board_list[i]
        constraint = Constraint('C_Row' + str(i), row)
        constraint.add_satisfying_tuples(all_satisfied)
        csp.add_constraint(constraint)
    # col
    for col_index in range(board_dim):
        col = []
        # add element at col_index in very row into col, to construct a column
        for row in board_list:
            col.append(row[col_index])
        constraint = Constraint('C_Col' + str(col_index), col)
        constraint.add_satisfying_tuples(all_satisfied)
        csp.add_constraint(constraint)
    return csp, board_list


def make_constraint(board_list, dataset, i, domain):
    operation_num = dataset[-1]
    target = dataset[-2]
    # construct variables list
    var_list = []
    for j in range(0, len(dataset) - 2):
        var_row = int(str(dataset[j])[0]) - 1
        var_col = int(str(dataset[j])[1]) - 1
        var = board_list[var_row][var_col]
        var_list.append(var)

    constraint = Constraint('C_' + str(i), var_list)
    valid = []

    if operation_num == 0:  # addition
        # addition: order does not affect sum
        all_combinations = list(itertools.combinations_with_replacement(domain, len(var_list)))

        for comb in all_combinations:
            if sum(comb) == target:
                valid.extend(list(itertools.permutations(comb, len(comb))))
                # valid.append(comb)
    elif operation_num == 1:  # subtraction
        all_combinations = list(itertools.product(domain, repeat=len(var_list)))
        for comb in all_combinations:
            if comb not in valid:
                difference = comb[0]
                for i in range(1, len(comb)):
                    difference -= comb[i]
                if difference == target:
                    valid.extend(list(itertools.permutations(comb, len(comb))))
    elif operation_num == 2:  # division
        all_combinations = list(itertools.product(domain, repeat=len(var_list)))
        for comb in all_combinations:
            if comb not in valid:
                quotient = comb[0]
                for i in range(1, len(comb)):
                    quotient /= comb[i]
                if quotient == target:
                    valid.extend(list(itertools.permutations(comb, len(comb))))

    elif operation_num == 3:  # multiplication
        all_combinations = list(itertools.combinations_with_replacement(domain, len(var_list)))

        for comb in all_combinations:
            product = 1
            for num in comb:
                product *= num
            if product == target:
                valid.extend(list(itertools.permutations(comb, len(comb))))
    valid = list(dict.fromkeys(valid))
    constraint.add_satisfying_tuples(valid)
    return constraint


def kenken_csp_model(kenken_grid):
    board_dim = kenken_grid[0][0]
    domain = list(range(1, board_dim + 1))  # construct domain for every variable
    csp, board_list = binary_ne_grid(kenken_grid)
    # csp, board_list = nary_ad_grid(kenken_grid)
    # add kenken constraints
    for i in range(1, len(kenken_grid)):
        dataset = kenken_grid[i]
        if len(dataset) == 2:
            target = dataset[-1]
            var_row = int(str(dataset[0])[0])
            var_col = int(str(dataset[0])[1])
            var = board_list[var_row][var_col]
            constraint = Constraint('C_' + str(i), var)
            constraint.add_satisfying_tuples([[target]])
            csp.add_constraint(constraint)
        elif len(dataset) > 2:
            constraint = make_constraint(board_list, dataset, i, domain)
            csp.add_constraint(constraint)
    return csp, board_list
