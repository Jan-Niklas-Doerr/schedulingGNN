import json
import os

import generator_layout
import generator_uncertainty
import generator_orderbook

class Generator:

    def __init__(self, path, exec_file, solver = 'Docplex.cp', time_limit = 300, layout_dict_default = {}, orderbook_dict_default = {}, uncertainty_dict_default = {}):
        assert os.path.exists(path), "Path does not exist"
        assert solver in ['Docplex.cp', 'OR-Tools'], "Solver not supported"

        self.solver_dict = {'time_limit': time_limit,
                            'solver': solver,
                            'exec_file': exec_file}
        self.path = path
        default_layout_dict = {'nr_stages': 4,
                'nr_products': 4,
                'prod_arrival_prob': [0.25, 0.25, 0.25, 0.25],
                'res_range': (2, 5),
                'skipping_prob': 0.0,
                'full_routing': True,
                'flexibility_target': 3.0,
                'processing_range': (10, 100),
                'initial_setup_range': (0, 0),
                'setup_range': (10, 100),
                'ratio_setup_processing': 0.5,
                'degree_of_unrelatedness': 0.5,
                'seed': None,
                'bottleneck_dict': None,
                'identical_machines': False
                }
        default_orderbook_dict = {'number_orders': 40,
                        'tightness_due_dates': 0,
                        'seed': None,
                        'single_orders': False,
                        'priorities': [1]
                        }
        default_uncertainty_dict = {'distribution': 'Exponential', 
                            'ratiominmean_setup': 0.8,
                            'ratiominmean_processing': 0.8
                            }
        
        assert all([ k in default_layout_dict.keys() for k in layout_dict_default.keys()]), "Invalid layout_dict keys"
        assert all([ k in default_orderbook_dict.keys() for k in orderbook_dict_default.keys()]), "Invalid orderbook_dict keys"
        assert all([ k in default_uncertainty_dict.keys() for k in uncertainty_dict_default.keys()]), "Invalid uncertainty_dict keys"
        
        self.default_layout_dict = {**default_layout_dict, **layout_dict_default}
        self.default_orderbook_dict = {**default_orderbook_dict, **orderbook_dict_default}
        self.default_uncertainty_dict = {**default_uncertainty_dict, **uncertainty_dict_default}

    def generate_batch(self, method, number, seeds = [], layout_dict = {}, uncertainty_dict = {}, orderbook_dict = {}, file_read = None, file_write = None, path = None):
        assert method in self.function.keys(), "Method not supported"
        assert number > 0, "Number of instances must be greater than 0"
        assert len(seeds) == 0 or len(seeds) == number, "Number of seeds must be equal to number of instances"
        if method not in ['instance', 'layout']:
            assert file_read, "File read must be specified"
        if file_write != None:
            assert len(file_write) == number, "Number of file writes must be equal to number of instances"

        assert all([ k in self.default_layout_dict.keys() for k in layout_dict.keys()]), "Invalid layout_dict keys"
        assert all([ k in self.default_orderbook_dict.keys() for k in orderbook_dict.keys()]), "Invalid orderbook_dict keys"
        assert all([ k in self.default_uncertainty_dict.keys() for k in uncertainty_dict.keys()]), "Invalid uncertainty_dict keys"
        
        if path != None:
            self.path = path

        function = {'instance': self.generate_instance,
                    'layout': self.generate_layout,
                    'uncertainty': self.generate_uncertainty,
                    'orderbook': self.generate_orderbook}

        for i in range(number):
            if file_write != None:
                file_writing = file_write[i]
            if seeds:
                layout_dict['seed'] = seeds[i]
                orderbook_dict['seed'] = seeds[i]
                if file_write == None: file_writing = str(seeds[i])
            else:
                if file_write == None: file_writing = str(i)
            function[method](layout_dict, file_writing, file_read, uncertainty_dict, orderbook_dict)

    def generate_instance(self, file_write, file_read = '', layout_dict = {}, uncertainty_dict = {}, orderbook_dict = {}, path = None):
        layout_dict = {**self.default_layout_dict, **layout_dict}
        uncertainty_dict = {**self.default_uncertainty_dict, **uncertainty_dict}
        orderbook_dict = {**self.default_orderbook_dict, **orderbook_dict}
        if orderbook_dict['single_orders']: assert len(set(layout_dict['prod_arrival_prob'])) == 1, "Unique orders only possible if all products have the same arrival probability"
        if path != None:
            self.path = path
        instance_dict = generator_layout.create_layout(**layout_dict, **self.solver_dict)
        generator_uncertainty.add_uncertainty(instance_dict, **uncertainty_dict)
        generator_orderbook.create_orderbook(instance_dict, **orderbook_dict)
        instance_dict["summary"] = {"layout_dict": layout_dict, "uncertainty_dict": uncertainty_dict, "orderbook_dict": orderbook_dict}
        self.save_to_json(instance_dict, file_write)

    def generate_layout(self, file_write, file_read = '', layout_dict = {}, uncertainty_dict = {}, orderbook_dict = {}, path = None):
        layout_dict = {**self.default_layout_dict, **layout_dict}
        if path != None:
            self.path = path
        instance_dict = generator_layout.create_layout(**layout_dict, **self.solver_dict)
        instance_dict["summary"] = {"layout_dict": layout_dict}
        self.save_to_json(instance_dict, file_write)

    def generate_uncertainty(self, file_write, file_read, layout_dict = {}, uncertainty_dict = {}, orderbook_dict = {}, path = None):
        with open(file_read, 'r') as f:
            instance_dict = json.load(f)
        uncertainty_dict = {**self.default_uncertainty_dict, **uncertainty_dict}
        if path != None:
            self.path = path
        generator_uncertainty.add_uncertainty(instance_dict, **uncertainty_dict)
        instance_dict["summary"]["uncertainty_dict"] = uncertainty_dict
        self.save_to_json(instance_dict, file_write)

    def generate_orderbook(self, file_write, file_read, layout_dict = {}, uncertainty_dict = {}, orderbook_dict = {}, path = None):
        with open(file_read, 'r') as f:
            instance_dict = json.load(f)
        orderbook_dict = {**self.default_orderbook_dict, **orderbook_dict}
        if path != None:
            self.path = path
        generator_orderbook.create_orderbook(instance_dict, **orderbook_dict)
        instance_dict["summary"]["orderbook_dict"] = orderbook_dict
        self.save_to_json(instance_dict, file_write)

    def save_to_json(self, file_dict, file_write):
        filename = self.path + file_write +'.json'
        with open(filename, 'w') as f:
            json.dump(file_dict, f, indent=4)