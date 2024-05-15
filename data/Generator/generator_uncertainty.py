import math

def get_distribution(type, mean, ratiominmean):
    if isinstance(mean, dict):
        if mean["type"] == "Nonexisting":
            return {"type": "Nonexisting"}
        else:
            mean = mean["mean"] 

    if mean == 0.0:
        return {"type": "Nonexisting"}
    else:
        distdict = {"type": type, "mean": mean}
        # add min
        distdict["min"] = ratiominmean*mean

        if type == "Exponential":
            parameters = [(1-ratiominmean)*mean]
            distdict["parameters"] = parameters
            distdict["median"] =  distdict["min"] + math.log(2) * parameters[0]

        elif type == "Uniform":
            parameters = [ratiominmean*mean - distdict["min"], (2-ratiominmean)*mean - distdict["min"]]
            distdict["parameters"] = parameters
            distdict["median"] = mean
            distdict["max"] = mean + (mean - distdict["min"])
        return distdict
    

def add_uncertainty(problem_instance, distribution, ratiominmean_setup, ratiominmean_processing):
    if distribution not in ["Exponential","Uniform"]:
        raise ValueError("either distribution is not supported (yet) or has a spelling mistake")
    if ratiominmean_setup < 0.0 or ratiominmean_setup > 1.0:
        raise ValueError("ratiominmean_setup must be between 0 and 1")
    if ratiominmean_processing < 0.0 or ratiominmean_processing > 1.0:
        raise ValueError("ratiominmean_processing must be between 0 and 1")
    # TODO allow different distributions and uncertainty for different stages/machines or even operations?

    for k, v in problem_instance['setup_time'].items():
        for kk, vv in v.items():
            for kkk, vvv in vv.items():
                problem_instance['setup_time'][k][kk][kkk] = get_distribution(distribution, vvv, ratiominmean_setup)
    for k, v in problem_instance['initial_setup'].items():
        for kk, vv in v.items():
            problem_instance['initial_setup'][k][kk] = get_distribution(distribution, vv, ratiominmean_setup)
    for k, v in problem_instance['processing_time'].items():
        for kk, vv in v.items():
            for kkk, vvv in vv.items():
                problem_instance['processing_time'][k][kk][kkk] = get_distribution(distribution, vvv, ratiominmean_processing)

    return problem_instance