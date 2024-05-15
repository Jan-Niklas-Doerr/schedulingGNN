import random

def create_orderbook(problem_instance, number_orders, tightness_due_dates, seed, single_orders, priorities):
    if seed is not None: random.seed = seed
    if single_orders: assert len(set(problem_instance['arrival_prob'])) == 1, "Unique orders only possible if all products have the same arrival probability"
    probabilities = problem_instance['arrival_prob']
    prob_vec = [p for p in probabilities.values()]
    prod_vec = [p for p in probabilities.keys()]

    if single_orders: assert len(prod_vec) == number_orders, "Unique orders only possible if number of orders equals number of products"

    if single_orders:
        products_drawn = prod_vec
    else:
        products_drawn = random.choices(prod_vec, weights = prob_vec, k = number_orders)

    orders = {}
    for i,j in enumerate(products_drawn):
        name = 'ORD_' + str(i+1).zfill(len(str(number_orders)))
        orders[name] = {'product' : j}

    problem_instance["orders"] = orders
    return problem_instance

def calc_min_due_date(instance, product):
    time = 0
    for o in instance['operations_per_product'][product]:
        min_time_op = min(
            instance['processing_time'][product][o][m]['mean'] + 
            instance['initial_setup'][m][product]['mean']
            if instance['initial_setup'][m][product]['type'] != 'Nonexisting'
            else instance['processing_time'][product][o][m]['mean']
            for m in instance['operation_machine_compatibility'][product][o]
        )
        time += min_time_op
    return time