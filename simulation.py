import numpy as np
import json
from queue import PriorityQueue


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

"""
class Product:

    # stage -> ("name", [resorces] )
    # operations = {}

    def __init__(self, name, due_date, operation_dict = None ):
        if operation_dict == None:
            self.operations = {}
        else:
            self.operations = operation_dict
        self.name = name
        self.due_date = due_date

    def add_operation(self,operation_name, stage,resources):
        
        try:
            self.operations[stage].append((operation_name,resources))
        except KeyError:
            self.operations[stage] = [(operation_name,resources)]

"""

def define_product_operations(products,data):

    new_products = {}
    for product in products:
        new_products[product] = {}
        for operation in data["operations_per_product"][product]:
            op_stage = data["product_operation_stage"][product][operation]
            op_resources = data["operation_machine_compatibility"][product][operation]
            try:
                new_products[product][op_stage].append((operation,op_resources))
            except KeyError:
                new_products[product][op_stage] = [(operation,op_resources)]
    
    return new_products

def define_resource_times(resources,products,data):
    for resource in resources:
        for product1, operations in products.items():
            resource.add_setup_time("", product1, 0)
            for product2 in products:
                resource.add_setup_time(product1, product2, data["setup_time"][resource.name][product1][product2].get("mean"))

            for product1_op_stage, product1_op_list in operations.items():
                for product1_op_name,product1_op_resources in product1_op_list:
                    if resource.name in product1_op_resources:
                        resource.add_processing_time(product1, product1_op_name, data["processing_time"][product1][product1_op_name][resource.name].get("mean") )
             
def initialize_stages(resources, stages):
    for resource in resources:
        stages[resource.stage-1].append(resource)
    return stages
    
def initialize_orders(orders, resources):
    return
    for resource in resources:
        invalid_orders = []

        while True:
            order = orders.get()
            
            


def form_orders(orders):
    ordered = PriorityQueue()
    for ord_name, args in orders.items():
        ordered.put((args["due_date"],ord_name,args["product"]))
    return ordered

with open('data/useCase_2_stages.json', 'r') as file:
    data = json.load(file)

# Accessing data

NR_STAGES = data["NrStages"]
stages = [ [] for _ in range(NR_STAGES) ]
operations = data["operations"]
products = [name for name in data["products"]]
resources = [Resource(name, data["resource_stage"][name]) for name in data["resources"]]

products = define_product_operations(products,data)

define_resource_times(resources,products,data)
stages = initialize_stages(resources, stages) # [[resource_list_of_stage_1],[#2], ... ]
orders = data["orders"] # priority queue (deadline, [ ord_num, product ] )

   
# SIMULATION
for p,op in products.items():
    print(p, op)

for r in resources:
    print(r.name, r.processing_times)
# process orders - simulate

action_list = PriorityQueue()
waiting_action_list = []

time = 0
orders = form_orders(orders)

initialize_orders(orders, stages[0])


    # take action(s)
    # validity check of action
    # update time
    