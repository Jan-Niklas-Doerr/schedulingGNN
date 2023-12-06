import numpy as np
import json
from queue import PriorityQueue
import plotly.figure_factory as ff
import random


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
    
    def __repr__(self):
        return "[ " + self.name + " " + str(self.stage) + " ]"


    def set_stage(self, stage):
        self.stage = stage

    def add_setup_time(self, product1, product2, time):
        self.setup_times[(product1,product2)] = time if time is not None else 0

    def add_processing_time(self, product, operation, time):
        self.processing_times[(product, operation)] = time

    def __lt__(self, obj):
        return self.name < obj.name
    
    def __le__(self, obj):
        return self.name <= obj.name

    def __eq__(self, obj):
        return self.name == obj.name

    def __ne__(self, obj):
        return self.name != obj.name

    def __gt__(self, obj):
        return self.name > obj.name

    def __ge__(self, obj):
        return self.name >= obj.name

class Order:


    def __init__(self, order_name, due_date, product_name ):
        
        self.order_name = order_name
        self.due_date = due_date
        self.product_name = product_name
        self.current_stage = 1

    def __repr__(self):
        return "[ " + self.order_name + " " + str(self.due_date) + " " + self.product_name + " ]"

    def increase_stage(self):
        self.current_stage = self.current_stage + 1

    def __lt__(self, obj):
        return self.due_date < obj.due_date # if self.due_date != obj.due_date else self.order_name < obj.order_name

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
    
def occupy_resource(time, resource, order, action_list,products, df):
    finish_time = time + resource.processing_times[(order.product_name, products[order.product_name][order.current_stage ][0][0])] + resource.setup_times[(resource.last_product,order.product_name)] 

    resource.is_occupied = True
    resource.last_product = order.product_name
    order.increase_stage()
    
    action_list.put((finish_time, order,resource ))

    df.append(dict(Task=resource.name, Start=time, Finish=finish_time, Resource=order.product_name))

def check_waiting_list(time, waiting_action_list, action_list,products, resource,df):

    """
    for i in range(len(waiting_action_list)):
        prod_t, resource_t = waiting_action_list[i]
        if resource.name in products[prod_t.product_name][prod_t.current_stage][0][1] :
            occupy_resource(time, resource, prod_t, action_list)
            resource.is_occupied = True
            resource_t.is_occupied = False
            waiting_action_list.pop(i)
            #print("We popped element, new length: ", len(waiting_action_list), "\n")
            break
    """

    #print("Checkin waiting list length: ", len(waiting_action_list))

    invalid_actions = []

    while not waiting_action_list.empty():
        prod_t, resource_t = waiting_action_list.get()
        if resource.name in products[prod_t.product_name][prod_t.current_stage ][0][1]:
            occupy_resource(time, resource, prod_t, action_list,products,df)

            #print("We popped element, new length: ", len(waiting_action_list), "\n")
            resource.is_occupied = True
            resource_t.is_occupied = False
            break
        else:
            invalid_actions.append((prod_t, resource_t))
    
    for remained_order in invalid_actions:
        waiting_action_list.put(remained_order)

def process_new_orders(time, orders, resources, products, action_list,df):

    # {product_name_1 : {stage_1 -> [("name", [resorce_names]), ...]} , 
    #      product_name_2 : {stage_1 -> [("name", [resorce_names]), ...]}  
    #   ... }

    # Order ==> priority queue (deadline, ord_num, product_name )

    for resource in resources:
        invalid_orders = []

        while not orders.empty():
            order = orders.get()

            if resource.name in products[order.product_name][order.current_stage ][0][1]:

                occupy_resource(time, resource, order, action_list,products,df)
                break
            else:
                invalid_orders.append(order)
        
        for remained_order in invalid_orders:
            orders.put(remained_order)

def process_new_orders_same_prod(time, orders, resources, products, action_list,df):

    # {product_name_1 : {stage_1 -> [("name", [resorce_names]), ...]} , 
    #      product_name_2 : {stage_1 -> [("name", [resorce_names]), ...]}  
    #   ... }

    # Order ==> priority queue (deadline, ord_num, product_name )

    for resource in resources:
        invalid_orders = []

        filled = False
        second_found = False
        second_best = ""
        while not orders.empty():
            order = orders.get()

            if resource.name in products[order.product_name][order.current_stage ][0][1] :
                if resource.last_product == order.product_name or resource.last_product == "":
                    occupy_resource(time, resource, order, action_list,products,df)
                    filled = True
                    if second_found :
                        invalid_orders.append(second_best)
                    break
                elif not second_found :
                    second_best = order
                    second_found = True
                else:
                    invalid_orders.append(order)
            else:
                invalid_orders.append(order)
        
        if not filled:
            occupy_resource(time, resource, second_best, action_list,products,df)
    
        for remained_order in invalid_orders:
            orders.put(remained_order)

def form_orders(orders):
    ordered = PriorityQueue()
    for order_name, args in orders.items():
        ordered.put(Order(order_name, args["due_date"], args["product"]))
    return ordered

def run(data, visualise, cleverInitialise, verbose):
    with open(data, 'r') as file:
        data = json.load(file)

    
    NR_STAGES = data["NrStages"]
    operations = data["operations"]
    df = []

    # {product_name_1 : {stage_1 -> ("name", [resorces], ...} , product_name_2 : {stage_1 -> ("name", [resorces], ...} }
    products = [name for name in data["products"]]
    products = define_product_operations(products,data)

    resources = [Resource(name, data["resource_stage"][name]) for name in data["resources"]]
    define_resource_times(resources,products,data)

    stages = initialize_stages(resources, [ [] for _ in range(NR_STAGES) ]) # [[resource_list_of_stage_1],[#2], ... ]
    orders = form_orders(data["orders"]) # priority queue (deadline, ord_num, product_name )
    
    # SIMULATION
    """
    for p,op in products.items():
        print(p, op)

    for r in resources:
        print(r.name, r.processing_times)
    """

    action_list = PriorityQueue()
    # [(finish_time, order, resource) ... ]

    waiting_action_list = PriorityQueue()

    time = 0
    success = 0


    process_new_orders(time, orders, stages[0], products, action_list,df) ## ACTION

    while not action_list.empty():

        time, prod, resource= action_list.get()
        # print(time, prod.product_name,prod.due_date, prod.order_name,prod.current_stage,resource.name, resource.stage)

        if prod.current_stage == NR_STAGES + 1 :

            resource.is_occupied = False
            if verbose:
                print("Order No: " , prod.order_name, " order product: ",prod.product_name , " is produced at time: ", time)
                print("Duedate was: ", prod.due_date, "\n")
            
            if prod.due_date >= time:
                success += 1

            check_waiting_list(time, waiting_action_list, action_list,products, resource, df) ## ACTION
            continue

        resource.is_occupied = False
        check_waiting_list(time, waiting_action_list, action_list,products, resource, df) ## ACTION


        assigned = False
        for resource_t in stages[prod.current_stage -1]:
            if not resource_t.is_occupied and resource_t.name in products[prod.product_name][prod.current_stage][0][1] :
                occupy_resource(time, resource_t, prod, action_list,products,df)  ## ACTION
                assigned = True
                break

        if not assigned:
            waiting_action_list.put((prod, resource))
        

        if resource.stage == 1 and not orders.empty():
            if cleverInitialise:
                process_new_orders_same_prod(time, orders, [resource], products, action_list, df) ## ACTION
            else:
                process_new_orders(time, orders, [resource], products, action_list, df) ## ACTION
        
    print("All products are produced at time: ", time)
    print("From total ", len(data["orders"]), " order, ", success, " was before their due dates. Success rate is: ", round((success / len(data["orders"]) * 100), 2) , "%" )

    if visualise:
            for task in df:
                assert 'Task' in task and 'Start' in task and 'Finish' in task and 'Resource' in task, "Data format error"
                assert task['Finish'] > task['Start'], "Finish time must be after Start time"

            # Create Gantt chart
            fig = ff.create_gantt(df, index_col='Resource',bar_width = 0.4, show_colorbar=True, group_tasks=True)
            fig.update_layout(xaxis_type='linear', autosize=False, width=800, height=400)

            # Show plot
            fig.show()