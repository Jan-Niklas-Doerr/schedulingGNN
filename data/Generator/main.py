import pathlib

import generator

if __name__ == "__main__":
    exec_file = "C:/Program Files/IBM/ILOG/CPLEX_Studio221/cpoptimizer/bin/x64_win64/cpoptimizer.exe"
    target = str(pathlib.Path(__file__).parent.parent.resolve()) + "/"


    # gen = generator.Generator(target, '', solver='OR-Tools')
    gen = generator.Generator(target, exec_file, solver='Docplex.cp')
    
    # generate layouts for
    stages = [1, 3]
    seeds = [11, 13]
    flexibility = [1.0, 3.0]
    setups = [0.0, 0.2]
    orderbook_seeds = [42, 43]
    
    for st in stages:
        for f in flexibility:
            for s in setups:
                for seed in seeds:
                    # res_range = (1, 1) if f == 1.0 else (2, 5)
                    # gen.generate_layout(f'Layout_f{int(f)}_s{int(s*5)}_st{st}_{seed}', layout_dict = {'nr_stages': st,
                    #                                         'res_range': res_range,
                    #                                         'flexibility_target': f,
                    #                                         'ratio_setup_processing': s,
                    #                                         'identical_machines': False,
                    #                                         'seed': seed})

                    # generate orderbooks

                    for o_seed in orderbook_seeds:
                        gen.generate_orderbook(f'Orderbook_f{int(f)}_s{int(s*5)}_st{st}_{seed}_{o_seed}', file_read =target + f'Layout_f{int(f)}_s{int(s*5)}_st{st}_{seed}.json', orderbook_dict = {'number_orders': 20, 'seed': o_seed})

    # read_path = str(dirname) + '/Instances/dpg_nosetup_singlestage_manyorders_e20.json'
    # gen.generate_uncertainty('dpg_nosetup_singlestage_manyorders_u20', read_path, uncertainty_dict={'distribution': 'Uniform', 'ratiominmean_setup': 0.8, 'ratiominmean_processing': 0.8})
    # gen.generate_orderbook('nosetup_singlestage_manyorders_e20', read_path, orderbook_dict={'number_orders': 120, 'tightness_due_dates': 0, 'seed': 42, 'single_orders': False, 'priorities': [1]})

