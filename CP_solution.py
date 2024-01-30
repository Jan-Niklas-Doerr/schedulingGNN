import docplex.cp.model as cp
import json
import pathlib
import os
import numpy as np
import random

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


def CPsolution(instance, objective, stoppingCriteria = "", saveGantt = False):
    """
    create CP model based on inputs and returns objective value (and solution?)
    """

    data = instance

    MACHINES = data["resources"]
    NB_MACHINES = len(MACHINES)

    tmpT = data["products"]
    NB_TYPES = len(tmpT)

    tmpJ = data["orders"].keys()
    NB_JOBS = len(tmpJ)

    tmpO = data["operations_per_product"]

    DUE_DATE_FLAG = False
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
            DUE_DATE_FLAG = True
            # raise Exception("due date flag is true")
            DUE_DATE.setdefault(a,round(data["orders"][a]["due_date"] * 1000))
        else:
            DUE_DATE.setdefault(a,1000000) # dummy value
        
        tmpType = data["orders"][a]["product"]
        TYPES_LOOKUP.append(tmpType)
        ops = data["operations_per_product"][tmpType]
        OPERATIONS.setdefault(a,ops)
        
        for b in ops:
            ELIGIBLE_MACHINES.setdefault(a, {}).setdefault(b, data["operation_machine_compatibility"][tmpType][b])

            for c in ELIGIBLE_MACHINES[a][b]:
                product = data["orders"][a]["product"]
                tmpTime = round(data["processing_time"][product][b][c]["mean"] * 1000) 
                OP_DURATIONS.setdefault(a, {}).setdefault(b, {}).setdefault(c, tmpTime)

    for m in MACHINES:
        TYPES.setdefault(m,[])
        for j in JOBS:
            tmpTO = JOBS.index(j) +1
            SETUPS.setdefault(m,{}).setdefault(tmpTO,[0 for i in range(NB_JOBS+1)])

    for a in tmpJ:
        tmpType = data["orders"][a]["product"]
        ops = data["operations_per_product"][tmpType]
    
        for b in ops:
            for c in ELIGIBLE_MACHINES[a][b]:

                tmpIndex = JOBS.index(a) +1
                TYPES[c].append(tmpIndex)
                for j in JOBS:
        
                    tmpFrom = JOBS.index(j) +1
                    jType = TYPES_LOOKUP[tmpFrom-1]
                    
                    if data["setup_time"][c][jType][tmpType]["type"] == "Nonexisting":
                        SETUPS[c][tmpIndex][tmpFrom] = 0
                    else:
                        SETUPS[c][tmpIndex][tmpFrom] = round(data["setup_time"][c][jType][tmpType]["mean"] * 1000)

    #-----------------------------------------------------------------------------
    # Build the model
    #-----------------------------------------------------------------------------

    # Create model
    mdl = cp.CpoModel()

    # Create one interval variable per job, operation
    Z_io =  { (i,o) : cp.interval_var(name='Z_{}{}'.format(i,o))
            for i,J in enumerate(JOBS) for o,O in enumerate(OPERATIONS[J])
            }

    # Create optional interval variable job operation machine if machine is eligible
    X_iom = { (i,o,MACHINES.index(M)) : cp.interval_var(name = 'X_{}{}{}'.format(i,o,MACHINES.index(M)), optional=True) 
            for i,J in enumerate(JOBS) for o,O in enumerate(OPERATIONS[J]) for M in ELIGIBLE_MACHINES[J][O]
            }

    # Create sequence variable 
    MCH_m = { (m) : cp.sequence_var([X_iom[a] for a in X_iom if a[2] == m], types = TYPES[M], name = 'MCH_{}'.format(m)) 
                for m,M in enumerate(MACHINES)
            }

    # ---------------------------------------------------------------------------------------------------
    # Constraints

    # TODO this does not work for parallel machines! 
    # probably because it sets not present interval variables to 0 which collides with the no overlap constraint
    for j,J in enumerate(JOBS):
        for o,O in enumerate(OPERATIONS[J]):
            for M in ELIGIBLE_MACHINES[J][O]:
                m = MACHINES.index(M)
                # if X_iom[j,o,m].is_present():
                st = cp.element(SETUPS[M][j+1],cp.type_of_prev(MCH_m[m], X_iom[j,o,m], 0, 0))
                pt = OP_DURATIONS[J][O][M]
                mdl.add(cp.length_of(X_iom[j,o,m],10000000000) >= st + pt)


    # Force each operation to start after the end of the previous
    mdl.add(cp.end_before_start(Z_io[i,o-1], Z_io[i,o]) for i,o in Z_io if 0<o)

    # Alternative constraints
    mdl.add(cp.alternative(Z_io[i,o], [X_iom[a] for a in X_iom if a[0:2]==(i,o)]) for i,o in Z_io)

    # Force no overlap for operations executed on a same machine
    mdl.add(cp.no_overlap(MCH_m[m]) for m in MCH_m)

    # add additional constraints same type with earlier due date needs to start before
    if DUE_DATE_FLAG:
        for i,i2 in enumerate(JOBS):
            for j,j2 in enumerate(JOBS):
                if TYPES_LOOKUP[i] == TYPES_LOOKUP[j]: 
                    if DUE_DATE[i2] < DUE_DATE[j2] or (DUE_DATE[i2] == DUE_DATE[j2] and i2 < j2):
                        mdl.add(cp.start_before_start(Z_io[i,0], Z_io[j,0]))

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
    objVec = [CMax/1000, TotalTardiness/1000] # TODO same order as in agent! divide time based objectives by 10000
    # TODO use objective vector!
    mdl.add(cp.minimize(cp.scal_prod(objective,objVec)))
    # mdl.add(cp.minimize((CMax) / 60))

    #-----------------------------------------------------------------------------
    # Solve the model and display the result
    #-----------------------------------------------------------------------------

    # Solve model 
    print('Solving model...')
    res = mdl.solve(TimeLimit= 600, # TODO dynamic stopping cirteria? and other params to tune solver!
                    agent = "local",
                    execfile="C:/Program Files/IBM/ILOG/CPLEX_Studio221/cpoptimizer/bin/x64_win64/cpoptimizer.exe")

    TmpObjective = res.get_objective_value()
    
    # Save Sequences
    SolSequences = {}
    for m,M in enumerate(MACHINES):
        #TODO right format of job and machine in INT! here or convert in agent? -> convert in agent via lookup table!
        SolSequences.setdefault(M,convert_to_order(str(res.get_var_solution(MCH_m[m])),JOBS,m)) 

    # TODO save sequence dict in JSON file!

    #-----------------------------------------------------------------------------
    # Show / Save Gantt
    #-----------------------------------------------------------------------------
    if saveGantt == True:
        colorMap = { p : ("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])) for p in tmpT }
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


    return TmpObjective, SolSequences # TODO return solution as well?


def optimal_solution(objective):

    """
    loop over all instances and according test samples in folder
    """
    basedir = str(pathlib.Path(__file__).parent.resolve()) + "/data"
    # get all files in directory excluding other directories
    files = [f for f in os.listdir(basedir) if os.path.isfile(os.path.join(basedir,f))]

    solutions = {}

    for filestr in files:
        with open(os.path.join(basedir,filestr)) as f:
            instance = json.load(f)

        obj, _ = CPsolution(instance, objective)
        solutions.setdefault(filestr,{"objective": obj})
 
        with open(os.path.join(basedir,"solutions.json"),'w') as f2:
            json.dump(solutions, f2, indent=4)

optimal_solution([1,0])
