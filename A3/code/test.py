elif operation_num == 2:  # division
# all_combinations = list(itertools.product(domain, repeat=len(var_list)))
all_combinations = list(itertools.combinations_with_replacement(domain, len(var_list)))
for comb in all_combinations:
    prod = 1.0
    for i in range(1, len(comb)):
        prod *= comb[i]
    for i in range(len(comb)):
        quotient = comb[i] ** 2 / prod
        if quotient == target:
            valid.extend(list(itertools.permutations(comb, len(comb))))