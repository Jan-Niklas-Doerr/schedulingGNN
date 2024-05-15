import json
import os
import pathlib
import random
import re
import warnings

import numpy as np
import docplex.cp.model as cp

# requires licence for cplex studio and a local installation 

# TODO 
def convert_to_order(seq, orderArray, machineNumber):
        ArrayA = seq.split(",")
        ArrayA[0] = ArrayA[0].split(' ')[1]
        ArrayA[-1] = ArrayA[-1][:-1]
        ArrayORD = []
        machineOffset = len(str(machineNumber))
        for a in ArrayA:
                number = a.split("_")[1]
                job = number[:-(machineOffset+1)]
                ArrayORD.append(orderArray[int(job)])
                
        return(ArrayORD)


#-----------------------------------------------------------------------------
# TODO Data Reader for new data format/instances!
#-----------------------------------------------------------------------------


def cp_solution(instance, objective, setups_coupled=True, time_limit=600, exp_measure="mean", saveGantt=False,
                precision=0.001, order_id_sorting=True, due_date_sorting=False, show_log=False):
    """
    create CP model based on inputs and returns objective value (and solution?)
    """
    if order_id_sorting and objective[1] > 0:
        warnings.warn("Order ID sorting can lead to inferior solutions wiht a tardiness objective!")
    assert order_id_sorting == False or due_date_sorting == False, "Only one sorting criteria can be used at a time!"
    data = instance

    MACHINES = data["resources"]
    NB_MACHINES = len(MACHINES)

    products = data["products"]
    NB_TYPES = len(products)

    tmpJ = data["orders"].keys()
    NB_JOBS = len(tmpJ)
    sorted_within_type = {product: [] for product in products}

    JOBS = []
    DUE_DATE = {}
    OPERATIONS = {}
    ELIGIBLE_MACHINES = {}
    OP_DURATIONS = {}

    TYPES = {}
    TYPES_LOOKUP = []
    SETUPS = {}
        
    for a in tmpJ:
        JOBS.append(a)
        if data["orders"][a]["due_date"] != None:
            # raise Exception("due date flag is true")
            DUE_DATE.setdefault(a,round(data["orders"][a]["due_date"] / precision))
        else:
            DUE_DATE.setdefault(a,1000000) # dummy value
        
        tmpType = data["orders"][a]["product"]
        sorted_within_type[tmpType].append(a)
        TYPES_LOOKUP.append(tmpType)
        ops = data["operations_per_product"][tmpType]
        OPERATIONS.setdefault(a,ops)
        
        for b in ops:
            ELIGIBLE_MACHINES.setdefault(a, {}).setdefault(b, data["operation_machine_compatibility"][tmpType][b])

            for c in ELIGIBLE_MACHINES[a][b]:
                product = data["orders"][a]["product"]
                tmpTime = round(data["processing_time"][product][b][c][exp_measure] / precision) 
                OP_DURATIONS.setdefault(a, {}).setdefault(b, {}).setdefault(c, tmpTime)

    for m in MACHINES:
        TYPES.setdefault(m,[])
        if setups_coupled:
            for j in JOBS:
                tmpTO = JOBS.index(j) +1
                SETUPS.setdefault(m,{}).setdefault(tmpTO,[0 for i in range(NB_JOBS+1)])
        else:
            SETUPS.setdefault(m,np.zeros((NB_JOBS +1, NB_JOBS +1)))

    for a in tmpJ:
        tmpType = data["orders"][a]["product"]
        ops = data["operations_per_product"][tmpType]
        tmpIndex = JOBS.index(a) + 1
        for b in ops:
            for c in ELIGIBLE_MACHINES[a][b]:
                # initial setup
                if setups_coupled:
                    SETUPS[c][tmpIndex][0] = 0
                else:
                    SETUPS[c][tmpIndex, 0] = 0
                # sequence dependent setup
                TYPES[c].append(tmpIndex)
                for j in JOBS:
                    tmpFrom = JOBS.index(j) + 1
                    jType = TYPES_LOOKUP[tmpFrom-1]
                    if data["setup_time"][c][jType][tmpType]["type"] == "Nonexisting":
                        if setups_coupled:
                            SETUPS[c][tmpIndex][tmpFrom] = 0
                        else:
                            SETUPS[c][tmpIndex, tmpFrom] = 0
                    else:
                        if setups_coupled:
                            SETUPS[c][tmpIndex][tmpFrom] = round(data["setup_time"][c][jType][tmpType][exp_measure] / precision)
                        else:
                            SETUPS[c][tmpIndex, tmpFrom] = round(data["setup_time"][c][jType][tmpType][exp_measure] / precision)
    #-----------------------------------------------------------------------------
    # Build the model
    #-----------------------------------------------------------------------------

    # Create model
    mdl = cp.CpoModel()

    # Create one interval variable per job, operation
    Z_io =  {(i,o): cp.interval_var(name='Z_{}{}'.format(i,o))
            for i,J in enumerate(JOBS) for o,O in enumerate(OPERATIONS[J])
            }

    if setups_coupled:
        # Create optional interval variable job operation machine if machine is eligible
        X_iom = {(i,o,MACHINES.index(M)): cp.interval_var(name = 'X_{}{}{}'.format(i,o,MACHINES.index(M)), optional=True) 
                for i,J in enumerate(JOBS) for o,O in enumerate(OPERATIONS[J]) for M in ELIGIBLE_MACHINES[J][O]
                }
    else:
        X_iom = {(i,o,MACHINES.index(M)): cp.interval_var(name = 'X_{}{}{}'.format(i,o,MACHINES.index(M)), optional=True, size = OP_DURATIONS[J][O][M]) 
            for i,J in enumerate(JOBS) for o,O in enumerate(OPERATIONS[J]) for M in ELIGIBLE_MACHINES[J][O]
            }

    # Create sequence variable 
    MCH_m = {(m): cp.sequence_var([X_iom[a] for a in X_iom if a[2] == m], types = TYPES[M], name = 'MCH_{}'.format(m)) 
                for m,M in enumerate(MACHINES)
            }

    # ---------------------------------------------------------------------------------------------------
    # Constraints

    if setups_coupled:
        # probably because it sets not present interval variables to 0 which collides with the no overlap constraint
        for j,J in enumerate(JOBS):
            for o,O in enumerate(OPERATIONS[J]):
                for M in ELIGIBLE_MACHINES[J][O]:
                    m = MACHINES.index(M)
                    # if X_iom[j,o,m].is_present():
                    st = cp.element(SETUPS[M][j+1],cp.type_of_prev(MCH_m[m], X_iom[j,o,m], 0, 0))
                    pt = OP_DURATIONS[J][O][M]
                    mdl.add(cp.length_of(X_iom[j,o,m],10000000000) >= st + pt)

        # Force no overlap for operations executed on a same machine
        mdl.add(cp.no_overlap(MCH_m[m]) for m in MCH_m)
    else:
        mdl.add(cp.no_overlap((MCH_m[m]), SETUPS[M], 1) for m,M in enumerate(MACHINES))

    # Force each operation to start after the end of the previous
    mdl.add(cp.end_before_start(Z_io[i,o-1], Z_io[i,o]) for i,o in Z_io if 0<o)

    # Alternative constraints
    mdl.add(cp.alternative(Z_io[i,o], [X_iom[a] for a in X_iom if a[0:2]==(i,o)]) for i,o in Z_io)


    # add additional constraints same type with earlier due date needs to start before
    if due_date_sorting:
        for i,i2 in enumerate(JOBS):
            for j,j2 in enumerate(JOBS):
                if TYPES_LOOKUP[i] == TYPES_LOOKUP[j]: 
                    if DUE_DATE[i2] < DUE_DATE[j2] or (DUE_DATE[i2] == DUE_DATE[j2] and i2 < j2):
                        mdl.add(cp.start_before_start(Z_io[i,0], Z_io[j,0]))

    if order_id_sorting:
        for p in products:
            for i,i2 in enumerate(sorted_within_type[p]):
                for j,j2 in enumerate(sorted_within_type[p][i+1:]):
                        index_i = JOBS.index(i2)
                        index_j = JOBS.index(j2)
                        mdl.add(cp.less_or_equal(cp.start_of(Z_io[index_i, 0]), cp.start_of(Z_io[index_j,0])))

    # TODO additional constraints from  Yunusoglu (2022) paper?
    # TODO check model? improve solving? -> recent Cp model and settings! cite it as benchmark with PI!

    # objective criteria

    # specify Tardiness
    Tardiness = [cp.max(0, cp.end_of(Z_io[i, len(OPERATIONS[J]) -1]) - (DUE_DATE[J])) for i,J in enumerate(JOBS)]

    TotalTardiness = cp.sum(Tardiness)

    # specifiy flow time
    Flowtime = [cp.end_of(Z_io[i,len(OPERATIONS[J])-1]) - cp.start_of(Z_io[i,0]) for i,J in enumerate(JOBS)]

    TotalFlow = cp.sum(Flowtime)

    # specifiy makespan
    CMax = cp.max(cp.end_of(Z_io[i,o]) for i,o in Z_io)

    # number of tardy Jobs
    NB_Tardy = cp.sum(i != 0 for i in Tardiness)

    # Minimize objective!
    # create objective vector
    objVec = [CMax * precision, TotalTardiness * precision] # TODO same order as in agent!
    # TODO use objective vector!
    mdl.add(cp.minimize(cp.scal_prod(objective,objVec)))
    # mdl.add(cp.minimize((CMax) / 60))

    #-----------------------------------------------------------------------------
    # Solve the model and display the result
    #-----------------------------------------------------------------------------

    # Solve model 
    print('Solving model...')
    res = mdl.solve(TimeLimit=time_limit, # TODO dynamic stopping cirteria? and other params to tune solver!
                    agent="local",
                    execfile="C:/Program Files/IBM/ILOG/CPLEX_Studio221/cpoptimizer/bin/x64_win64/cpoptimizer.exe",
                    trace_log=show_log)
    if res:
        TmpObjective = res.get_objective_value()
        
        # Save Sequences
        SolSequences = {}
        for m,M in enumerate(MACHINES):
            #TODO right format of job and machine in INT! here or convert in agent? -> convert in agent via lookup table!
            SolSequences.setdefault(M,convert_to_order(str(res.get_var_solution(MCH_m[m])),JOBS,m)) 

        curve = []
        log = res.get_solver_log()
        for line in log.split('\n'):
            # if line != None and line != '':
            if "*" in line:
                tmp = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                curve.append([float(tmp[2]), float(tmp[0])])
        #-----------------------------------------------------------------------------
        # Show / Save Gantt
        #-----------------------------------------------------------------------------
        if saveGantt == True:
            colorMap = {p: ("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])) for p in products}
            # Draw solution
            import docplex.cp.utils_visu as visu
            if res and visu.is_visu_enabled():
            # Draw solution
                visu.timeline('Solution for test')
                visu.panel('Machines')
                for m in range(NB_MACHINES):
                    visu.sequence(name='RES_' + str(m+1))
                    for a in X_iom:
                        if a[2]==m:
                            itv = res.get_var_solution(X_iom[a])
                            # print(TYPES_LOOKUP[a[0]], ": ", itv)
                            if itv.is_present():
                                visu.interval(itv, colorMap[TYPES_LOOKUP[a[0]]], JOBS[a[0]])
                visu.show()
                #TODO save Gantt!
        return TmpObjective, SolSequences, curve # TODO return solution as well?


def expected_sequence(instance_names, objective, exp_measure="mean", setups_coupled=True, **kwargs):
    """
    loop over all instances and according test samples in folder
    """
    assert exp_measure in ["mean", "median"], "Invalid measure for expected value!"
    
    if objective == [0, 1]:
        obj = "tardiness"
    elif objective == [1, 0]:
        obj = "makespan"

    for instance_file in instance_names:
        file_name = instance_file + ".json"
        with open(os.path.join("data/", file_name)) as f:
            instance = json.load(f)

        obj, sequence, curve = cp_solution(instance, objective, setups_coupled=setups_coupled, exp_measure=exp_measure, **kwargs)
        print("Objective Value: ", obj)
        dict_seq = {}
        dict_seq.setdefault('sequence', sequence)
        dict_seq.setdefault('objective', {}).setdefault('exp_objective', obj)
        dict_seq['objective'].setdefault('obj_curve', curve)

        solution_file = instance_file + "_sol.json"

        with open(os.path.join("data/", solution_file), 'w') as f2:
            json.dump(dict_seq, f2,indent=4)

def compare_sol_gantt(basedir, filestr, sample, objective):

    with open(os.path.join(basedir,filestr)) as f:
        instance = json.load(f)
    cp_solution(instance, objective, saveGantt=True)


    
# b = str(pathlib.Path(__file__).parent.resolve()) + "/Benchmark_Kurz/base_setting/data_ii"
# f = "llllll-0_ii_Expo5.json"
# s = "2.json"

# comparsolgantt(b,f,s,[1])
if __name__ == "__main__":
    expected_sequence(["No_flex_No_setup_5_stages", "No_flex_Yes_setup_5_stages"], [1, 0], exp_measure="mean")
# compare_sol_gantt(str(pathlib.Path(__file__).parent.resolve()) + "/Benchmark_Doerr/Instances/", "default_problem.json", "", [0, 1])