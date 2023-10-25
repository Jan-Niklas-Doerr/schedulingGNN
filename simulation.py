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

class Order:


    def __init__(self, order_name, due_date, product_name ):
        
        self.order_name = order_name
        self.due_date = due_date
        self.product_name = product_name
        self.current_stage = 1

    def increase_stage(self):
        self.current_stage = self.current_stage + 1

    def __lt__(self, obj):
        return self.due_date < obj.due_date

    def __le__(self, obj):
        return self.due_date <= obj.due_date

    def __eq__(self, obj):
        return self.due_date == obj.due_date

    def __ne__(self, obj):
        return self.due_date != obj.due_date

    def __gt__(self, obj):
        return self.due_date > obj.due_date

    def __ge__(self, obj):
        return self.due_date >= obj.due_date

   
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
    

def occupy_resource(time, resource, order, action_list):
    finish_time = time + resource.processing_times[(order.product_name, products[order.product_name][order.current_stage][0][0])] + resource.setup_times[(resource.last_product,order.product_name)] 

    resource.is_occupied = True
    resource.last_product = order.product_name
    order.increase_stage()
    
    action_list.put((finish_time, order,resource ))

def process_new_orders(time, orders, resources, products, action_list):

    # {product_name_1 : {stage_1 -> [("name", [resorce_names]), ...]} , 
    #      product_name_2 : {stage_1 -> [("name", [resorce_names]), ...]}  
    #   ... }

    # Order ==> priority queue (deadline, ord_num, product_name )

    for resource in resources:
        invalid_orders = []

        while not orders.empty():
            order = orders.get()
            if resource.name in products[order.product_name][order.current_stage][0][1]:

                occupy_resource(time, resource, order, action_list)
                # set action
                break
            else:
                invalid_orders.append(order)
        
        for remained_order in invalid_orders:
            orders.put(remained_order)


def form_orders(orders):
    ordered = PriorityQueue()
    for order_name, args in orders.items():
        ordered.put(Order(order_name, args["due_date"], args["product"]))
    return ordered

with open('data/useCase_2_stages.json', 'r') as file:
    data = json.load(file)

# Accessing data

NR_STAGES = data["NrStages"]
operations = data["operations"]

# {product_name_1 : {stage_1 -> ("name", [resorces], ...} , product_name_2 : {stage_1 -> ("name", [resorces], ...} }
products = [name for name in data["products"]]
products = define_product_operations(products,data)

resources = [Resource(name, data["resource_stage"][name]) for name in data["resources"]]
define_resource_times(resources,products,data)

stages = initialize_stages(resources, [ [] for _ in range(NR_STAGES) ]) # [[resource_list_of_stage_1],[#2], ... ]
orders = form_orders(data["orders"]) # priority queue (deadline, ord_num, product_name )
remaining_orders = data["orders"]
   
# SIMULATION
"""
for p,op in products.items():
    print(p, op)

for r in resources:
    print(r.name, r.processing_times)
"""
# process orders - simulate

action_list = PriorityQueue()
# [(finish_time, order, resource) ... ]

waiting_action_list = []

time = 0

process_new_orders(time, orders, stages[0], products, action_list)

while not action_list.empty():

    # time, prod, resource= action_list.get()
    # print(time, prod.product_name,prod.due_date, prod.order_name,prod.current_stage,resource.name)

    time, prod, resource = action_list.get()

    resource.is_occupied = False

    assigned = False
    for resource in stages[prod.current_stage]:
        if resource.name in products[prod.product_name][prod.current_stage][0][1]:
            occupy_resource(time, resource, prod, action_list)
            assigned = True
            break

    if not assigned:
        waiting_action_list.append((time, prod, resource))
    
    if resource.stage == 0:
        process_new_orders(time, orders, [resource], products, action_list)




    # take action(s)
    # validity check of action
    # update time
    
print("All products are successfully produced at time " + time)