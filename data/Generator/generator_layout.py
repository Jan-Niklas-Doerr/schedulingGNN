import numpy as np
import random

def get_instance(products, prod_arrival_prob, nr_stages, res_range, flexibility_score,
                 product_operation_stage, operations_per_product, full_routing,
                 degree_of_unrelatedness, domain_processing, domain_initial_setup,
                 domain_setup, setup_ratio, bottleneck_dict, time_limit, solver, exec_file, identical_machines, epsilon = 0.00):
    
    d = [i for i in range(res_range[0], res_range[1]+1)]
    product_stages = {p : {s : 1 if s in product_operation_stage[p].values() else 0
                           for s in range(1, nr_stages+1)}  for p in products}
    product_nb_stages = {p : sum(product_stages[p].values()) for p in products}
    possible_machines = ['RES_{}_{}'.format(str(s).zfill(len(str(nr_stages))),str(m).zfill(len(str(res_range[1]))))
                          for s in range(1,nr_stages+1) for m in range(1,res_range[1]+1)]
    possible_machines_stage = {s : ['RES_{}_{}'.format(str(s).zfill(len(str(nr_stages))),str(m).zfill(len(str(res_range[1]))))
                                    for m in range(1,res_range[1]+1)] 
                               for s in range(1,nr_stages+1)}
    product_machine_operation = {p : {m : [o for o in operations_per_product[p] 
                                           if m in possible_machines_stage[product_operation_stage[p][o]]]
                                           for m in possible_machines} for p in products}

    domain_processing_base_list = [i for i in range(domain_processing[0], domain_processing[1]+1)] # TODO adapt?!
    domain_processing_list = [0] + [i for i in range(domain_processing[0], domain_processing[1]+1)]
    domain_initial_setup_list = [i for i in range(domain_initial_setup[0], domain_initial_setup[1]+1)]
    domain_setup_list = [0] + [i for i in range(domain_setup[0], domain_setup[1]+1)]

    stage_machines = {s : 0 for s in range(1,nr_stages+1)}
    if nr_stages != 1:
        s = random.choice(range(1, nr_stages +1))
        tmp_machine_max = min(int(flexibility_score +1), res_range[1])
        tmp_machine_min = max(int(flexibility_score -1), res_range[0])
        stage_machines[s] = random.randint(tmp_machine_min, tmp_machine_max)

    if bottleneck_dict is None:
        bottleneck_dict = {s : 1 for s in range(1, nr_stages+1)}

    # randomize some machine assigments         
    if not full_routing:
        product_machine_assigments = {}
        # TODO randomize
        # random.randint()

    # TODO randomize some entries
    init_setup = {m : {p : -1 for p in products} for m in possible_machines}
    # init_setup = {m : {p : random.randint(domain_initial_setup_list[0], domain_initial_setup_list[-1]) for p in products} for m in possible_machines}

    if solver == 'Docplex.cp': 
        import docplex.cp.model as cp
        mdl = cp.CpoModel()
        # variables
        
        M_s =  {s: cp.integer_var(name='M_{}'.format(s), domain = d)
                for s in range(1,nr_stages+1) 
                }
        Y_m = {m: cp.binary_var(name='Y_{}'.format(m))
                for m in possible_machines
                }

        X_mi = {(m, i): cp.binary_var(name='X_{}_{}'.format(m, i)) 
                for m in possible_machines 
                for i in products
                }
        
        P_Base_io = {(i,o): cp.integer_var(name='P_{}_{}'.format(i,o), domain = domain_processing_base_list)
                    for i in products
                    for o in operations_per_product[i]
                    }
        
        P_iom =  {(i,o,m): cp.integer_var(name='P_{}_{}_{}'.format(i,o,m), domain = domain_processing_list)
                for i in products
                for o in operations_per_product[i]
                for m in possible_machines_stage[product_operation_stage[i][o]]
                    }
        
        S_initial_mi =  {(m,i): cp.integer_var(name='S_{}_{}'.format(m,i), domain = domain_initial_setup_list)
                        for i in products
                        for m in possible_machines
                        }
        
        S_mij =  {(m,i,j): cp.integer_var(name='S_{}_{}_{}'.format(m,i,j), domain = domain_setup_list)
                for i in products
                for j in products
                for m in possible_machines
                }
        
        # W_m = {m : cp.integer_var(name='W_{}'.format(m)) for m in possible_machines}

        # constraints

        if full_routing:
            mdl.add(cp.equal(product_stages[i][s] * Y_m[m], X_mi[m,i])
                            for i in products for s in range(1, nr_stages+1) for m in possible_machines_stage[s])
        else:
            mdl.add(cp.greater(cp.sum(product_stages[i][s] * cp.sum(Y_m[m] for m in possible_machines_stage[s]) for s in range(1, nr_stages+1)),
                                cp.sum(X_mi[m,i] for m in possible_machines))
                                for i in products)
            
        # only machines active that have at least one product assigned
        mdl.add(cp.less_or_equal(Y_m[m], cp.sum(X_mi[m,i] for i in products)) for m in possible_machines)

        # only assigment to machine if present
        mdl.add(cp.greater_or_equal(Y_m[m], X_mi[m,i]) for m in possible_machines for i in products)

        # enforce random machine drawn
        mdl.add(cp.equal(M_s[s], stage_machines[s])  for s in range(1, nr_stages+1) if stage_machines[s] > 0) 
        
        # existing machines on each stage has to equal to number of machines on stage
        mdl.add(cp.equal(cp.sum(Y_m[m] for m in possible_machines_stage[s]), M_s[s]) for s in range(1, nr_stages+1))

        # symmetry breaker
        mdl.add(cp.greater_or_equal(Y_m[m], Y_m[possible_machines_stage[s][i+1]]) for s in range(1, nr_stages+1) for i,m in enumerate(possible_machines_stage[s][:-1]))

        # range of assigments
        # >=1 if present at stage
        mdl.add(cp.greater_or_equal(cp.sum(X_mi[m, i] for m in possible_machines_stage[s]), product_stages[i][s]) for s in range(1, nr_stages+1) for i in products)
        # # <= M_s * present at stage | done in contraint above
        # mdl.add(cp.less_or_equal(cp.sum(X_mi[m, i] for m in possible_machines_stage[s]), M_s[s] * product_stages[i][s]) for s in range(1, nr_stages+1) for i in products)
        
        # hard constraint,^
        
        mdl.add(cp.equal(
            cp.sum(X_mi[m,i] for m in possible_machines), int(flexibility_score * product_nb_stages[i]))
            for i in products)

        # TODO add epsilon for constraint
        # mdl.add(cp.greater_or_equal(
        #     cp.sum(X_mi[m,i] for m in possible_machines),
        #     (1 - epsilon) * flexibility_score * product_nb_stages[i])
        #     for i in products)
        
        # mdl.add(cp.less_or_equal(
        #     cp.sum(X_mi[m,i] for m in possible_machines),
        #     (1 + epsilon) * flexibility_score * product_nb_stages[i])
        #     for i in products)

        
        # relate processing time to base for same operations
        mdl.add(cp.less_or_equal(
            P_Base_io[i,o] * (1 - degree_of_unrelatedness) * X_mi[m,i] , P_iom[i,o,m])
            for i in products for o in operations_per_product[i]
            for m in possible_machines_stage[product_operation_stage[i][o]]
            )
        mdl.add(cp.greater_or_equal(
            P_Base_io[i,o] * (1 + degree_of_unrelatedness) * X_mi[m,i], P_iom[i,o,m])
            for i in products for o in operations_per_product[i]
            for m in possible_machines_stage[product_operation_stage[i][o]]
            )
        
        # non present products get zero processing time
        mdl.add(cp.less_or_equal(P_iom[i,o,m], domain_processing[1] * X_mi[m,i]) for i in products for o in operations_per_product[i]
                for m in possible_machines_stage[product_operation_stage[i][o]]
                )
        
        # setups
        if setup_ratio == 0:
            mdl.add(cp.equal(S_mij[m,i,j], 0) for m in possible_machines for i in products for j in products)

        else:
            # enforcing zero for same product
            mdl.add(cp.equal(S_mij[m,i,j], 0) for m in possible_machines for i in products for j in products if i == j)

            # enforcing zero for not present products
            mdl.add(cp.less_or_equal(S_mij[m,i,j], domain_setup[1] * X_mi[m,i] * X_mi[m,j] ) for m in possible_machines for i in products for j in products if i != j)

            # asymmetry using big M with two different values
            mdl.add(cp.diff(S_mij[m,i,j] +  (domain_setup[1] + 1) * (1 - X_mi[m,i]), S_mij[m,j,i] + (domain_setup[1] + 2) * (1 - X_mi[m,j]))
                    for m in possible_machines for i in products for j in products if i != j)

            # TODO is this needed? could have zero setup between products!?
            mdl.add(cp.diff(S_mij[m,i,j], (X_mi[m,i] + X_mi[m,j] - 2)) for m in possible_machines for i in products for j in products if i != j)

            # triangle inequality
            mdl.add(cp.less_or_equal(S_mij[m,i,j], S_mij[m,i,k] + S_mij[m,k,j] + domain_setup[1] * (1 - X_mi[m,k])) for m in possible_machines 
                    for i in products for j in products for k in products
                    if i != j and i != k and j != k)

            # initial setup
            mdl.add(cp.equal(S_initial_mi[m,i], init_setup[m][i] * X_mi[m,i]) for m in possible_machines for i in products if init_setup[m][i] >= 0)

            # setup ratio == average setuptime / avg processing time
            mdl.add(cp.equal(
                cp.sum(P_iom[i,o,m] for i in products for o in operations_per_product[i] for m in possible_machines_stage[product_operation_stage[i][o]]) * setup_ratio /
                cp.sum(X_mi[m,i] for m in possible_machines for i in products),
                (cp.sum(S_mij[m,i,j] for m in possible_machines for i in products for j in products) + 
                cp.sum(S_initial_mi[m,i] for m in possible_machines for i in products)) /
                (cp.sum(cp.sum(X_mi[m,i] for i in products)**2 + cp.sum(X_mi[m,i] for i in products) for m in possible_machines))
                ))
        

        if identical_machines:
            for i in products:
                for o in operations_per_product[i]:
                    stage = product_operation_stage[i][o]
                    for m in possible_machines_stage[stage]:
                        mdl.add(cp.equal(P_iom[i,o,m], P_Base_io[i,o] * X_mi[m,i]))
                        for m2 in possible_machines_stage[stage]:
                            if m2 != m:
                                mdl.add(cp.equal(S_initial_mi[m,i], S_initial_mi[m2,i]))
                                for j in products:
                                    mdl.add(cp.equal(S_mij[m,i,j], S_mij[m2,i,j]))

            # TODO add constraint for same average workload, i.e. processingtimes / number machines sould be equal
            workload = {s: (bottleneck_dict[s] * cp.sum(prod_arrival_prob[i] * P_Base_io[i, o]
                                                        for i in products for o in operations_per_product[i] if product_operation_stage[i][o] == s) / M_s[s])
                                                        for s in range(1, nr_stages+1)}

            max_workload = cp.max(workload[s] for s in range(1, nr_stages+1))
            min_workload = cp.min(workload[s] for s in range(1, nr_stages+1))
            mdl.add(cp.less_or_equal(max_workload, (1 + epsilon) * min_workload))
        else:
            # workload of each stage
            workload = {s: bottleneck_dict[s] * cp.sum(
                                cp.sum(prod_arrival_prob[i] * (P_iom[i, o, m] +
                                                                    (
                                    (cp.sum(S_mij[m,j,i] for j in products) + S_initial_mi[m,i]) /
                                    (cp.sum(X_mi[m,j] for j in products) + 1) 
                                )
                                ) /
                                (1 - Y_m[m] + cp.sum(X_mi[r,i] for r in possible_machines_stage[product_operation_stage[i][o]]))
                                for i in products for o in product_machine_operation[i][m]) 
                            for m in possible_machines_stage[s])
                            / M_s[s]
                        for s in range(1, nr_stages+1)
                        }

            max_workload = cp.max(workload[s] for s in range(1, nr_stages+1))
            min_workload = cp.min(workload[s] for s in range(1, nr_stages+1))
            # min_workload = cp.min(Workload[s] + (max_workload * (1 - Y_m[m])) for m in possible_machines)

            mdl.add(cp.less_or_equal(max_workload, (1 + epsilon) * min_workload))
            # mdl.add(cp.minimize(violation))
        res = mdl.solve(TimeLimit= time_limit, 
                        agent = "local",
                        execfile= exec_file,
                        LogVerbosity = "Quiet",
                        trace_log = False
                        )
        if res:
            nb_machines = np.zeros(nr_stages)
            for n in M_s:
                nb_machines[n-1] = res[M_s[n]]

            routing = {}
            if not full_routing:
                for x in X_mi:
                    if res[X_mi[x]] == 1:
                        for o in product_machine_operation[x[1]][x[0]]:
                            routing.setdefault(x[1], {}).setdefault(o, []).append(str(x[0]))

            processing_times = {}
            setup_times = {}
            initial_setup_times = {}
            for p in P_iom:
                if res[X_mi[p[2], p[0]]] == 1:
                    processing_times.setdefault(p[0], {}).setdefault(p[1], {}).setdefault(p[2], res[P_iom[p]])
            for s in S_mij:
                if res[X_mi[s[0], s[1]]] == 1 and res[X_mi[s[0], s[2]]] == 1: #TODO might have to remove to not break sim
                    setup_times.setdefault(s[0], {}).setdefault(s[1], {}).setdefault(s[2], res[S_mij[s]])
            for i in S_initial_mi:
                if res[X_mi[i[0], i[1]]] == 1:
                    initial_setup_times.setdefault(i[0], {}).setdefault(i[1], res[S_initial_mi[i]])

            workload_machine = {m: cp.sum(prod_arrival_prob[i] * (res[P_iom[i,o,m]] +
                                                                (
                                (cp.sum(res[S_mij[m,i,j]] for j in products) + res[S_initial_mi[m,i]]) /
                                (cp.sum(res[X_mi[m,j]] for j in products) + 1) 
                                )
                                ) /
                                (1 - res[Y_m[m]] + cp.sum(res[X_mi[r,i]] for r in possible_machines_stage[product_operation_stage[i][o]]))
                                for i in products for o in product_machine_operation[i][m]) 
                            for m in possible_machines if res[Y_m[m]] == 1
                            }
            max_workload = max(workload_machine.values())
            norm_workload = {key : value / max_workload for key, value in workload_machine.items()}
            return nb_machines, routing, processing_times, setup_times, initial_setup_times, \
                    norm_workload, workload_machine, epsilon
        else:
            # if epsilon < 0.05:
            #     epsilon += 0.01
            #     get_instance(products, prod_arrival_prob, nr_stages, res_range, flexibility_score,
            #                 product_operation_stage, operations_per_product, full_routing,
            #                 degree_of_unrelatedness, domain_processing, domain_initial_setup,
            #                 domain_setup, setup_ratio, bottleneck_dict = bottleneck_dict, epsilon = epsilon)
            # else:
            failed_constraints = mdl.refine_conflict()
            print(failed_constraints)
            raise ValueError("No solution found, unfavorable parameters?!")
        
    elif solver == 'OR-Tools':
        from ortools.sat.python import cp_model as cp

        mdl = cp.CpModel()
        # variables
        M_s =  {s: mdl.NewIntVar(d[0], d[-1], name='M_{}'.format(s))
                for s in range(1,nr_stages+1) 
                }
        Y_m = {m: mdl.new_bool_var(name='Y_{}'.format(m))
                for m in possible_machines
                }

        X_mi = {(m, i): mdl.new_bool_var(name='X_{}_{}'.format(m, i)) 
                for m in possible_machines 
                for i in products
                }
        
        P_Base_io = {(i,o): mdl.NewIntVar(domain_processing_base_list[0], domain_processing_base_list[-1], name='P_{}_{}'.format(i,o))
                    for i in products
                    for o in operations_per_product[i]
                    }
        
        P_iom =  {(i,o,m): mdl.NewIntVar(domain_processing_list[0], domain_processing_list[-1], name='P_{}_{}_{}'.format(i,o,m))
                for i in products
                for o in operations_per_product[i]
                for m in possible_machines_stage[product_operation_stage[i][o]]
                    }
        
        S_initial_mi =  {(m,i): mdl.NewIntVar(domain_initial_setup_list[0], domain_initial_setup_list[-1], name='S_{}_{}'.format(m,i))
                        for i in products
                        for m in possible_machines
                        }
        
        S_mij =  {(m,i,j): mdl.NewIntVar(domain_setup_list[0], domain_setup_list[-1], name='S_{}_{}_{}'.format(m,i,j))
                for i in products
                for j in products
                for m in possible_machines
                }

        if full_routing:
            for i in products:
                for s in range(1, nr_stages+1):
                    for m in possible_machines_stage[s]:
                        mdl.add(product_stages[i][s] * Y_m[m] == X_mi[m,i])
        else:
            for i in products:
                mdl.add(sum(product_stages[i][s] * sum(Y_m[m] for m in possible_machines_stage[s]) for s in range(1, nr_stages+1)) >
                                sum(X_mi[m,i] for m in possible_machines))
   
            
        # only machines active that have at least one product assigned
        for m in possible_machines:
            mdl.add(Y_m[m] <= sum(X_mi[m,i] for i in products))

        # only assigment to machine if present
        for m in possible_machines:
            for i in products:
                mdl.add(Y_m[m] >= X_mi[m,i])

        # enforce random machine drawn
        for s in range(1, nr_stages+1):
            if stage_machines[s] > 0:
                mdl.add(M_s[s] == stage_machines[s])
        
        # existing machines on each stage has to equal to number of machines on stage
        for s in range(1, nr_stages+1):
            mdl.add(sum(Y_m[m] for m in possible_machines_stage[s]) == M_s[s])

        # symmetry breaker
        for s in range(1, nr_stages+1):
            for i,m in enumerate(possible_machines_stage[s][:-1]):
                mdl.add(Y_m[m] >= Y_m[possible_machines_stage[s][i+1]])

        # needs to be assignet to at least one machine if present
        for s in range(1, nr_stages+1):
            for i in products:
                mdl.add(sum(X_mi[m, i] for m in possible_machines_stage[s]) >= product_stages[i][s])

        # hard constraint flexibility score
        for i in products:
            mdl.add(sum(X_mi[m,i] for m in possible_machines) == int(flexibility_score * product_nb_stages[i]))

        # relate processing time to base for same operations
        helper_processing = {(i,o,m):  mdl.NewIntVar(0, domain_processing_list[-1], name='HP_{}_{}_{}'.format(i,o,m) ) for i in products for o in operations_per_product[i] for m in possible_machines_stage[product_operation_stage[i][o]]}
        precision = 100
        dou_lb = int(precision * (1 - degree_of_unrelatedness))
        dou_ub = int(precision * (1 + degree_of_unrelatedness))
        for i in products:
            for o in operations_per_product[i]:
                for m in possible_machines_stage[product_operation_stage[i][o]]:
                    mdl.AddMultiplicationEquality(helper_processing[i,o,m], [P_Base_io[i,o], X_mi[m,i]])
                    mdl.add(helper_processing[i,o,m] * dou_lb <= P_iom[i,o,m] * precision)
                    mdl.add(helper_processing[i,o,m] * dou_ub >= P_iom[i,o,m] * precision)
                    
        # non present products get zero processing time
        for i in products:
            for o in operations_per_product[i]:
                for m in possible_machines_stage[product_operation_stage[i][o]]:
                    mdl.add(P_iom[i,o,m] <= domain_processing[1] * X_mi[m,i])

        # setups
        # enforcing zero for same product
        for m in possible_machines:
            for i in products:
                mdl.add(S_mij[m,i,i] == 0)
        
        # enforcing zero for not present products
        helper_setup = {(m,i,j): mdl.NewIntVar(0, 1, name='HS_{}_{}_{}'.format(m,i,j)) for m in possible_machines for i in products for j in products if i != j}
        for m in possible_machines:
            for i in products:
                for j in products:
                    if i != j:
                        mdl.AddMultiplicationEquality(helper_setup[m,i,j], [X_mi[m,i],X_mi[m,j]])
                        mdl.add(S_mij[m,i,j] <= domain_setup[1] * helper_setup[m,i,j])
        
        # asymmetry using big M with two different values
        helper_assymetry = {(m,i,j): mdl.NewIntVar(0, (domain_setup_list[-1] + 2), name='HA_{}_{}_{}'.format(m,i,j))
                            for m in possible_machines for i in products for j in products if i != j}
        for m in possible_machines:
            for digit,i in enumerate(products):
                for j in products[digit:-1]:
                    if i != j:
                        mdl.add(helper_assymetry[m,i,j] == S_mij[m,i,j] + (domain_setup[1] + 1) * (1 - X_mi[m,i]))
                        mdl.add(helper_assymetry[m,j,i] == S_mij[m,j,i] + (domain_setup[1] + 2) * (1 - X_mi[m,j]))
                        mdl.AddAllDifferent([helper_assymetry[m,i,j], helper_assymetry[m,j,i]])

        # TODO is this needed? could have zero setup between products!?
        helper_no_zero = {(m,i,j): mdl.NewIntVar(0, domain_setup_list[-1], name='HNZ_{}_{}_{}'.format(m,i,j)) for m in possible_machines for i in products for j in products if i != j}
        for m in possible_machines:
            for i in products:
                for j in products:
                    if i != j:
                        mdl.add(helper_no_zero[m,i,j] == S_mij[m,i,j] - (X_mi[m,i] + X_mi[m,j] - 2))
                        mdl.AddAllDifferent([helper_no_zero[m,i,j], 0])

        # triangle inequality
        for m in possible_machines:
            for i in products:
                for j in products:
                    for k in products:
                        if i != j and i != k and j != k:
                            mdl.add(S_mij[m,i,j] <= S_mij[m,i,k] + S_mij[m,k,j] + domain_setup[1] * (1 - X_mi[m,k]))

        # initial setup
        for m in possible_machines:
            for i in products:
                if init_setup[m][i] >= 0:
                    mdl.add(S_initial_mi[m,i] == init_setup[m][i] * X_mi[m,i])

        # setup ratio == average setuptime / avg processing time

        average_setup = mdl.NewIntVar(0, domain_setup_list[-1] * precision, name='AS')
        total_setup = mdl.NewIntVar(0, domain_setup_list[-1] * (len(products)**2 + len(products)), name='TS')
        setup_denom = mdl.NewIntVar(1, len(products) * len(possible_machines), name='SD')
        
        number_prod_machine = {m: mdl.NewIntVar(0, len(products), name='NPM_{}'.format(m)) for m in possible_machines}
        number_setups_machine = {m: mdl.NewIntVar(0, len(products)**2, name='NSM_{}'.format(m)) for m in possible_machines}
        
        average_processing = mdl.NewIntVar(0, domain_processing_list[-1] * precision, name='AP')
        total_processing = mdl.NewIntVar(0, domain_processing_list[-1] * len(products) * max(len(operations_per_product[i]) for i in products) , name='TP')
        processing_denom = mdl.NewIntVar(sum(len(operations_per_product[i]) for i in products), len(products) * len(possible_machines), name='PD')
        
        mdl.Add(total_processing == sum(P_iom[i,o,m] for i in products for o in operations_per_product[i]
                                    for m in possible_machines_stage[product_operation_stage[i][o]]))
        mdl.Add(processing_denom == sum(X_mi[m,i] for m in possible_machines for i in products))
        
        mdl.AddDivisionEquality(average_processing, total_processing * precision, processing_denom)

        for m in possible_machines:
            mdl.add(number_prod_machine[m] == sum(X_mi[m,i] for i in products))
            mdl.add_multiplication_equality(number_prod_machine[m], [number_setups_machine[m], number_setups_machine[m]])

        mdl.Add(total_setup == sum(S_mij[m,i,j] for m in possible_machines for i in products for j in products) 
                + sum(S_initial_mi[m,i] for m in possible_machines for i in products))
        
        mdl.Add(setup_denom == sum(number_prod_machine[m] + number_setups_machine[m] for m in possible_machines))

        mdl.AddDivisionEquality(average_setup, total_setup * precision, setup_denom)

        setup_ratio_precision = int(precision * setup_ratio)
        mdl.add(average_setup * precision == average_processing * setup_ratio_precision)


        # workload of each stage
        avg_setup_p_m = {(m,i): mdl.NewIntVar(0, 10000000, name='ASPM_{}_{}'.format(m,i)) for m in possible_machines for i in products}
        total_setup_p_m = {(m,i): mdl.NewIntVar(0, 100000, name='TSPM_{}_{}'.format(m,i)) for m in possible_machines for i in products}
        denom_setup_p_m = {(m,i): mdl.NewIntVar(1, 100000, name='DSPM_{}_{}'.format(m,i)) for m in possible_machines for i in products}

        workload_p_m = {(m,i): mdl.NewIntVar(0, 10000000, name='WPM_{}_{}'.format(m,i)) for m in possible_machines for i in products}
        max_pos_workload_p_m = {(m,i): mdl.NewIntVar(0, 100000, name='MWPM_{}_{}'.format(m,i)) for m in possible_machines for i in products}
        denom_workload_p_m = {(m,i): mdl.NewIntVar(1, 100000, name='DWPM_{}_{}'.format(m,i)) for m in possible_machines for i in products}

        workload_stage = {s: mdl.NewIntVar(0, 1000000000, name='W_{}'.format(s)) for s in range(1, nr_stages+1)}
        avg_workload_stage = {s: mdl.NewIntVar(0, 1000000000, name='W_{}'.format(s)) for s in range(1, nr_stages+1)}
        max_workload = mdl.NewIntVar(0, 1000000000, name='MW')
        min_workload = mdl.NewIntVar(0, 1000000000, name='mW')

        for m in possible_machines:
            for i in products:
                mdl.Add(total_setup_p_m[m,i] == sum(S_mij[m,j,i] for j in products) + S_initial_mi[m,i])
                mdl.Add(denom_setup_p_m[m,i] == sum(X_mi[m,j] for j in products) + 1)
                mdl.AddDivisionEquality(avg_setup_p_m[m,i], total_setup_p_m[m,i] * precision, denom_setup_p_m[m,i])

                for o in product_machine_operation[i][m]:
                    mdl.Add(max_pos_workload_p_m[m,i] == P_iom[i,o,m] * precision + avg_setup_p_m[m,i])
                    mdl.Add(denom_workload_p_m[m,i] == 1 - Y_m[m] + sum(X_mi[r,i] for r in possible_machines_stage[product_operation_stage[i][o]]))
                    mdl.AddDivisionEquality(workload_p_m[m,i],
                                            max_pos_workload_p_m[m,i] * int(prod_arrival_prob[i] * precision),
                                            denom_workload_p_m[m,i]
                    )

        for s in range(1, nr_stages+1):
            tmp_bottleneck = bottleneck_dict[s] * precision
            mdl.Add(workload_stage[s] == tmp_bottleneck * sum(workload_p_m[m,i] for m in possible_machines_stage[s] for i in products))
            mdl.AddDivisionEquality(avg_workload_stage[s], workload_stage[s], M_s[s])
        
        mdl.AddMaxEquality(max_workload, [avg_workload_stage[s] for s in range(1, nr_stages+1)])
        mdl.AddMinEquality(min_workload, [avg_workload_stage[s] for s in range(1, nr_stages+1)])
        epsilon_precision = int(precision * (1 + epsilon))
        mdl.add(precision * max_workload <= epsilon_precision * min_workload)

        solver = cp.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(mdl)

        if status == cp.OPTIMAL or status == cp.FEASIBLE:

            for i in products:
                for m in possible_machines:
                    print('product: ', i, 'machine: ', m)
                    print('machine active: ', solver.Value(Y_m[m]))
                    print('product assigned: ', solver.Value(X_mi[m,i]))
                    print('avg_setup: ', solver.Value(avg_setup_p_m[m,i]))
                    print('max_workload: ', solver.Value(max_pos_workload_p_m[m,i]))

            for s in range(1, nr_stages+1): print('workload: ', solver.Value(workload_stage[s]))
            for s in range(1, nr_stages+1): print(solver.Value(avg_workload_stage[s]))
            print(solver.Value(max_workload), solver.Value(min_workload))

            nb_machines = np.zeros(nr_stages)
            for n in M_s.keys():
                nb_machines[n-1] = solver.Value(M_s[n])

            routing = {}
            if not full_routing:
                for x in X_mi.keys():
                    if solver.Value(X_mi[x]) == 1:
                        for o in product_machine_operation[x[1]][x[0]]:
                            routing.setdefault(x[1], {}).setdefault(o, []).append(str(x[0]))

            processing_times = {}
            setup_times = {}
            initial_setup_times = {}
            for p in P_iom.keys():
                if solver.Value(X_mi[p[2], p[0]]) == 1:
                    processing_times.setdefault(p[0], {}).setdefault(p[1], {}).setdefault(p[2], solver.Value(P_iom[p]))
            for s in S_mij.keys():
                if solver.Value(X_mi[s[0], s[1]]) == 1 and solver.Value(X_mi[s[0], s[2]]) == 1: #TODO might have to remove to not break sim
                    setup_times.setdefault(s[0], {}).setdefault(s[1], {}).setdefault(s[2], solver.Value(S_mij[s]))
            for i in S_initial_mi.keys():
                if solver.Value(X_mi[i[0], i[1]]) == 1:
                    initial_setup_times.setdefault(i[0], {}).setdefault(i[1], solver.Value(S_initial_mi[i]))

            workload_machine = {m: sum(prod_arrival_prob[i] * (solver.Value(P_iom[i,o,m]) +
                                                                (
                                (sum(solver.Value(S_mij[m,i,j]) for j in products) + solver.Value(S_initial_mi[m,i])) /
                                (sum(solver.Value(X_mi[m,j]) for j in products) + 1) 
                                )
                                ) /
                                (1 - solver.Value(Y_m[m]) + sum(solver.Value(X_mi[r,i]) for r in possible_machines_stage[product_operation_stage[i][o]]))
                                for i in products for o in product_machine_operation[i][m]) 
                            for m in possible_machines if solver.Value(Y_m[m]) == 1
                            }
            max_workload = max(workload_machine.values())
            norm_workload = {key : value / max_workload for key, value in workload_machine.items()}
            return nb_machines, routing, processing_times, setup_times, initial_setup_times, \
                    norm_workload, workload_machine, epsilon
        else:
            print(mdl.Validate())
            if status == cp.INFEASIBLE:
                print("Infeasible")
            elif status == cp.MODEL_INVALID:
                print("Model invalid")
            elif status == cp.UNKNOWN:
                print("Unknown")
            
            # if epsilon < 0.05:
            #     epsilon += 0.01
            #     get_instance(products, prod_arrival_prob, nr_stages, res_range, flexibility_score,
            #                 product_operation_stage, operations_per_product, full_routing,
            #                 degree_of_unrelatedness, domain_processing, domain_initial_setup,
            #                 domain_setup, setup_ratio, bottleneck_dict = bottleneck_dict, epsilon = epsilon)
            # else:
            raise ValueError("No solution found, unfavorable parameters?!")



def create_layout(nr_stages:int, nr_products:int, prod_arrival_prob:list, res_range:tuple, skipping_prob:float,
                    full_routing:bool, flexibility_target:float, 
                    processing_range:tuple, initial_setup_range:tuple,
                    setup_range:tuple, ratio_setup_processing:float,
                    degree_of_unrelatedness:float, seed, bottleneck_dict, time_limit, solver, exec_file, identical_machines
                    ):
    """
    Creates an instance of the flow shop problem with the given parameters.
    """
    assert len(prod_arrival_prob) == nr_products, "Arrival probabilities do not match number of products"
    assert all([p != 0 for p in prod_arrival_prob]), "Arrival probabilities of zero is not supported"
    assert sum(prod_arrival_prob) == 1, "Arrival probabilities do not sum to 1"
    assert nr_stages > 0, "Invalid number of stages"
    assert nr_products > 0, "Invalid number of products"
    assert 0 <= res_range[0] <= res_range[1], "Invalid range for number of machines"
    assert 0 <= skipping_prob <= 1, "Invalid skipping probability"
    assert 0 <= flexibility_target, "Invalid flexibility target"
    assert 0 <= degree_of_unrelatedness <= 1, "Invalid degree of unrelatedness"
    assert 0 <= processing_range[0] <= processing_range[1], "Invalid range for processing times"
    assert 0 <= initial_setup_range[0] <= initial_setup_range[1], "Invalid range for initial setup times"
    assert 0 <= setup_range[0] <= setup_range[1], "Invalid range for setup times"
    assert 0 <= ratio_setup_processing, "Invalid ratio of setup to processing times"


    prod_arrival_prob = {'PRODUCT_{}'.format(str(i+1).zfill(len(str(nr_products)))) : p for i, p in enumerate(prod_arrival_prob)}
    # generate Layout
    # Resources
    resources = []
    stage_resource = {}
    resource_stage = {}

    if seed:
        random.seed(seed)

    products = ["PRODUCT_" + str(i+1).zfill(len(str(nr_products))) for i in range(nr_products)]
    operations = set()
    product_operation_stage = {}
    operations_per_product = {k :[] for k in products}
    operation_machine_compatibility = {}
    for p in products:
        stages_present = 0
        for i in range(1, nr_stages+1):
            if random.random() >= skipping_prob or (stages_present == 0 and i == nr_stages):
                stages_present += 1
                op_str = 'OP_{}'.format(str(i).zfill(len(str(nr_stages))))
                operations.add(op_str)
                product_operation_stage.setdefault(p, {}).setdefault(op_str, i)
                operations_per_product[p].append(op_str)
    operations = list(operations)

    number_machine_stages, routing, processing_times, setup_times \
        ,initial_setups, norm_workloads, workloads, epsilon = \
            get_instance(products, prod_arrival_prob, nr_stages, res_range, flexibility_target,
                         product_operation_stage, operations_per_product, full_routing,
                         degree_of_unrelatedness, processing_range, initial_setup_range,
                         setup_range, ratio_setup_processing, bottleneck_dict, time_limit, solver,
                         exec_file, identical_machines)
    
    # get_instance(products, prod_arrival_prob, nr_stages, res_range, flexibility_score, product_operation_stage, operations_per_product, full_routing,
    # degree_of_unrelatedness, domain_processing, domain_initial_setup, domain_setup, setup_ratio):
    
    for i in range(1, nr_stages+1):
        tmp_res = []
        for j in range(int(number_machine_stages[i-1])):
            tmp_name = 'RES_{}_{}'.format(str(i).zfill(len(str(nr_stages))),str(j+1).zfill(len(str(res_range[1]))))
            resources.append(tmp_name)
            tmp_res.append(tmp_name)
            resource_stage[tmp_name] = i
        stage_resource[i] = tmp_res

    if full_routing:
        for p in products:
            for op_str in operations_per_product[p]:
                operation_machine_compatibility.setdefault(p, {}).setdefault(op_str, stage_resource[product_operation_stage[p][op_str]])
    else:
        operation_machine_compatibility = routing

    # generate dict
    file_dict = {}
    file_dict['resources'] = resources
    file_dict['resource_stage'] = resource_stage
    file_dict['stage_resource'] = stage_resource
    file_dict['NrStages'] = nr_stages
    file_dict['products'] = products
    file_dict['operations'] = operations
    file_dict['product_operation_stage'] = product_operation_stage
    file_dict['operations_per_product'] = operations_per_product
    file_dict['operation_machine_compatibility'] = operation_machine_compatibility
    file_dict['processing_time'] = processing_times
    file_dict['setup_time'] = setup_times
    file_dict['initial_setup'] = initial_setups
    file_dict['arrival_prob'] = prod_arrival_prob

    file_dict['norm_workloads'] = norm_workloads
    file_dict['workloads'] = workloads
    file_dict['flexibility_score'] = flexibility_target
    file_dict['epsilon'] = epsilon
    file_dict['seed_layout'] = seed


    return file_dict


if __name__ == "__main__":
    create_layout(nr_stages= 4, nr_products = 4, prod_arrival_prob = [0.3, 0.3, 0.2, 0.2], res_range = (2,5), skipping_prob = 0.2,
                    full_routing = False, flexibility_target = 2.0, seed = 42
                    )