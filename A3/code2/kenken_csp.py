#Look for #IMPLEMENT tags in this file.
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
    ##IMPLEMENT
    length = kenken_grid[0][0]
    domain = []
    var_list = []
    all_var = []
    con_list = []

    for i in range(1, length + 1):
        domain.append(i)
    satisfying_tuples = list(itertools.permutations(domain, 2))
    for i in range(length):
        lst = []
        for j in range(length):
            var = Variable(str(i + 1) + str(j + 1), domain)
            all_var.append(var)
            lst.append(var)
            for l in range(0, j):
                constraint = Constraint(str(i) + str(j) + str(l), [var, lst[l]])
                constraint.add_satisfying_tuples(satisfying_tuples)
                con_list.append(constraint)

        var_list.append(lst)
    for j in range(length):
        for i in range(length):
            for l in range(0, i):
                lst = []
                lst.append(var_list[i][j])
                lst.append(var_list[l][j])
                constraint = Constraint(str(i) + str(j) + str(l), lst)
                constraint.add_satisfying_tuples(satisfying_tuples)
                con_list.append(constraint)

    csp = CSP("binary", all_var)
    for i in con_list:
        csp.add_constraint(i)
    return csp, var_list


def nary_ad_grid(kenken_grid):
    ##IMPLEMENT
    length = kenken_grid[0][0]
    domain = []
    var_list = []
    all_var = []
    con_list = []

    for i in range(1, length + 1):
        domain.append(i)
    satisfying_tuples = list(itertools.permutations(domain))
    for i in range(length):
        lst = []
        for j in range(length):
            var_name = str(i + 1) + str(j + 1)
            var = Variable(var_name, domain)
            all_var.append(var)
            lst.append(var)
        constraint = Constraint('h' + str(j), lst)
        constraint.add_satisfying_tuples(satisfying_tuples)
        con_list.append(constraint)
        var_list.append(lst)
    for j in range(length):
        lst = []
        for i in range(length):
            lst.append(var_list[i][j])
        constraint = Constraint('v' + str(j), lst)
        constraint.add_satisfying_tuples(satisfying_tuples)
        con_list.append(constraint)
    csp = CSP("nary", all_var)
    for i in con_list:
        csp.add_constraint(i)
    return csp, var_list


def kenken_csp_model(kenken_grid):
    ##IMPLEMENT
    csp, var_list = binary_ne_grid(kenken_grid)
    length = kenken_grid[0][0]
    domain = []
    kenken_grid_t = kenken_grid[1:]
    for i in range(1, length + 1):
        domain.append(i)

    for i in kenken_grid_t:
        # print(i)
        cage_length = len(i)
        op = i[-1]
        goal = i[-2]
        variables = i[:-2]
        num_var = len(variables)
        variables_type = []
        for j in variables:
            j = str(j)
            variables_type.append(var_list[int(j[0]) - 1][int(j[1]) - 1])
        constraint = Constraint("cage_" + str(i), variables_type)
        # print("i",i,op)
        if (op == 0):
            # plus
            satisfying_tuples = []
            for j in list(itertools.combinations_with_replacement(domain, num_var)):
                num = 0
                for k in j:
                    num += k
                if (num == goal):
                    for z in list(itertools.permutations(j)):
                        satisfying_tuples.append(z)

            constraint.add_satisfying_tuples(satisfying_tuples)



        elif (op == 1):
            # subtract
            satisfying_tuples = []
            for j in list(itertools.combinations_with_replacement(domain, num_var)):
                num = 0

                for k in j:
                    num += k
                for k in j:
                    if (k * 2 - num == goal):
                        for z in list(itertools.permutations(j)):
                            satisfying_tuples.append(z)

            constraint.add_satisfying_tuples(satisfying_tuples)


        elif (op == 2):
            # divide
            satisfying_tuples = []
            for j in list(itertools.combinations_with_replacement(domain, num_var)):
                num = 1
                for k in j:
                    num *= k
                for k in j:
                    if (k ** 2 / num == goal):
                        for z in list(itertools.permutations(j)):
                            satisfying_tuples.append(z)

            constraint.add_satisfying_tuples(satisfying_tuples)



        elif (op == 3):
            # multiply
            satisfying_tuples = []
            for j in list(itertools.combinations_with_replacement(domain, num_var)):
                num = 1
                for k in j:
                    num *= k
                if (num == goal):
                    for z in list(itertools.permutations(j)):
                        satisfying_tuples.append(z)

            constraint.add_satisfying_tuples(satisfying_tuples)

        csp.add_constraint(constraint)
    return csp, var_list

