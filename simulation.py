import numpy as np
import json


class Resource:

    # (product_name, operation) -> time
    # processing_times = {}
    
    # (product_name, product_name) -> time
    # setup_times = {}

    def __init__(self, name="", stage=0):
        self.name = name
        self.stage = stage
        self.processing_times = {}
        self.setup_times = {}
        self.last_product = "" # Update when product process
        self.is_occupied = False
    
    def set_stage(self, stage):
        self.stage = stage

    def add_setup_time(self, product1, product2, time):
        self.setup_times[(product1,product2)] = time if time is not None else 0

    def add_processing_time(self, product, operation, time):
        self.processing_times[(product, operation)] = time

class Product:

    # stage -> ("name", [resorces] )
    # operations = {}

    def __init__(self, name, operation_dict = None ):
        if operation_dict == None:
            self.operations = {}
        else:
            self.operations = operation_dict
        self.name = name
    def add_operation(self,operation_name, stage,resources):
        
        try:
            self.operations[stage].append((operation_name,resources))
        except KeyError:
            self.operations[stage] = [(operation_name,resources)]
    


with open('data/useCase_2_stages.json', 'r') as file:
    data = json.load(file)

# Accessing data

NR_STAGES = data["NrStages"]
operations = data["operations"]
products = [Product(name) for name in data["products"]]

resources = [Resource(name, data["resource_stage"][name]) for name in data["resources"]]

for product in products:
    for operation in data["operations_per_product"][product.name]:
        op_stage = data["product_operation_stage"][product.name][operation]
        op_resources = data["operation_machine_compatibility"][product.name][operation]
        product.add_operation(operation, op_stage, op_resources)


for resource in resources:
    for product1 in products:
        resource.add_setup_time("", product1, 0)
        for product2 in products:
            resource.add_setup_time(product1.name, product2.name, data["setup_time"][resource.name][product1.name][product2.name].get("mean"))

        for product1_op_stage, product1_op_list in product1.operations.items():
            for product1_op_name,product1_op_resources in product1_op_list:
                if resource.name in product1_op_resources:
                    resource.add_processing_time(product1.name, product1_op_name, data["processing_time"][product1.name][product1_op_name][resource.name].get("mean") )
                
# SIMULATION

# process orders - simulate
