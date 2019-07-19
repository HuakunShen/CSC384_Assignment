# Look for #IMPLEMENT tags in this file. These tags indicate what has
# to be implemented.

import random

'''
This file will contain different variable ordering heuristics to be used within
bt_search.

var_ordering == a function with the following template
    var_ordering(csp)
        ==> returns Variable 

    csp is a CSP object---the heuristic can use this to get access to the
    variables and constraints of the problem. The assigned variables can be
    accessed via methods, the values assigned can also be accessed.

    var_ordering returns the next Variable to be assigned, as per the definition
    of the heuristic it implements.

val_ordering == a function with the following template
    val_ordering(csp,var)
        ==> returns [Value, Value, Value...]
    
    csp is a CSP object, var is a Variable object; the heuristic can use csp to access the constraints of the problem, and use var to access var's potential values. 

    val_ordering returns a list of all var's potential values, ordered from best value choice to worst value choice according to the heuristic.

'''


def ord_mrv(csp):
    unassigned_vars = csp.get_all_unasgn_vars()
    min_dom_size = float('inf')
    mrv_var = None
    for var in unassigned_vars:
        if var.cur_domain_size() < min_dom_size:
            mrv_var = var
            min_dom_size = var.cur_domain_size()
    return mrv_var


def val_lcv(csp, var):
    data = []
    for val in var.cur_domain():
        data.append([val, 0])
    for constraint in csp.get_cons_with_var(var):
        for dataset in data:
            val = dataset[0]
            var.assign(val)
            for other_var in constraint.get_unasgn_vars():
                for other_var_val in other_var.cur_domain():
                    if not constraint.has_support(other_var, other_var_val):
                        dataset[1] += 1
            var.unassign()
    data.sort(key=lambda tup: tup[1])
    result = []
    for dataset in data:
        result.append(dataset[0])
    return result
