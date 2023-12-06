import numpy as np
import json
from queue import PriorityQueue
import random
import plotly.figure_factory as ff


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
        self.last_product = "" 
        self.is_occupied = False
        self.free_at = 0
    
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
        self.resource = None

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

class Env:

    def __init__(self, data, visualise ):
        with open(data, 'r') as file:
            self.data = json.load(file)

        # Accessing data

        self.visualise = visualise
        self.df = []

        self.NR_STAGES = self.data["NrStages"]
        self.operations = self.data["operations"]

        # {product_name_1 : {stage_1 -> ("name", [resorces], ...} , product_name_2 : {stage_1 -> ("name", [resorces], ...} }
        self.products = [name for name in self.data["products"]]
        self.define_product_operations()

        self.resources = {name: Resource(name, self.data["resource_stage"][name]) for name in self.data["resources"]}
        self.define_resource_times()

        self.stages = self.initialize_stages( [ [] for _ in range(self.NR_STAGES) ]) # [[resource_list_of_stage_1],[#2], ... ]
        self.orders = {order_name:Order(order_name, args["due_date"], args["product"]) for order_name, args in self.data["orders"].items()}

        self.action_list = PriorityQueue()
        # [(finish_time, order, resource) ... ]

        self.waiting_action_list = PriorityQueue()

        self.time = 0
        self.success = 0
        self.last_job = 0
        self.alive = True

        self.possible_actions = {order.order_name: self.products[order.product_name][order.current_stage][0][1] for order in self.orders.values()}

    def define_product_operations(self):

        new_products = {}
        for product in self.products:
            new_products[product] = {}
            for operation in self.data["operations_per_product"][product]:
                op_stage = self.data["product_operation_stage"][product][operation]
                op_resources = self.data["operation_machine_compatibility"][product][operation]
                try:
                    new_products[product][op_stage].append((operation,op_resources))
                except KeyError:
                    new_products[product][op_stage] = [(operation,op_resources)]
        
        self.products = new_products

    def define_resource_times(self):
        for resource in self.resources.values():
            for product1, operations in self.products.items():
                resource.add_setup_time("", product1, 0)
                for product2 in self.products:
                    resource.add_setup_time(product1, product2, self.data["setup_time"][resource.name][product1][product2].get("mean"))

                for product1_op_stage, product1_op_list in operations.items():
                    for product1_op_name,product1_op_resources in product1_op_list:
                        if resource.name in product1_op_resources:
                            resource.add_processing_time(product1, product1_op_name, self.data["processing_time"][product1][product1_op_name][resource.name].get("mean") )
                
    def initialize_stages(self, stages):
        for resource in self.resources.values():
            stages[resource.stage-1].append(resource)
        self.stages = stages


    def terminate(self):
        print("All products are produced at time: ", self.time)
        print("From total ", len(self.data["orders"]), " order, ", self.success, " was before their due dates. Success rate is: ", round((self.success / len(self.data["orders"]) * 100), 2) , "%" )
        self.alive = False

        """
        for task in self.df:
            task['Start'] = f'Time {task["Start"]}'
            task['Finish'] = f'Time {task["Finish"]}'

            if task['Finish'] <= task['Start']:
                print(task['Start'], task['Finish'])
"""
        if self.visualise:
            for task in self.df:
                assert 'Task' in task and 'Start' in task and 'Finish' in task and 'Resource' in task, "Data format error"
                assert task['Finish'] > task['Start'], "Finish time must be after Start time"


            r = lambda: random.randint(0,255)             
            colors = ['#%02X%02X%02X' % (r(),r(),r())]              
            for i in range(1, len(self.orders)+1):                                   
                colors.append('#%02X%02X%02X' % (r(),r(),r()))

            # Create Gantt chart
            fig = ff.create_gantt(self.df, index_col='Resource',bar_width = 0.4, show_colorbar=True, group_tasks=True)
            fig.update_layout(xaxis_type='linear', autosize=False, width=800, height=400)

            # Show plot
            fig.show()

    def reset(self):
        self.alive = True
        print("reseted")

    def occupy_resource(resource, order):
            finish_time = self.time + resource.processing_times[(order.product_name, self.products[order.product_name][order.current_stage ][0][0])] + resource.setup_times[(resource.last_product,order.product_name)] 

            resource.is_occupied = True
            resource.last_product = order.product_name
            order.increase_stage()
            
            # KEEP OCCUPIATION TIME IN RESOURCE
            self.action_list.put((finish_time, order,resource ))

    def check_waiting_list(time, waiting_action_list, action_list, resource):

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
                occupy_resource(time, resource, prod_t, action_list)

                #print("We popped element, new length: ", len(waiting_action_list), "\n")
                resource.is_occupied = True
                resource_t.is_occupied = False
                break
            else:
                invalid_actions.append((prod_t, resource_t))
        
        for remained_order in invalid_actions:
            waiting_action_list.put(remained_order)

    def process_new_orders(time, orders, resources, products, action_list):

        # {product_name_1 : {stage_1 -> [("name", [resorce_names]), ...]} , 
        #      product_name_2 : {stage_1 -> [("name", [resorce_names]), ...]}  
        #   ... }

        # Order ==> priority queue (deadline, ord_num, product_name )

        for resource in resources:
            invalid_orders = []

            while not orders.empty():
                order = orders.get()

                if resource.name in products[order.product_name][order.current_stage ][0][1]:

                    occupy_resource(time, resource, order, action_list)
                    break
                else:
                    invalid_orders.append(order)
            
            for remained_order in invalid_orders:
                orders.put(remained_order)

    def process_new_orders_same_prod(time, orders, resources, products, action_list):

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
                        occupy_resource(time, resource, order, action_list)
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
                occupy_resource(time, resource, second_best, action_list)
        
            for remained_order in invalid_orders:
                orders.put(remained_order)

    def form_orders(orders):
        ordered = PriorityQueue()
        for order_name, args in orders.items():
            ordered.put(Order(order_name, args["due_date"], args["product"]))
        return ordered

    def step(self, action):
        order_name, resource_name = action
        order = self.orders[order_name]
        resource = self.resources[resource_name]

        if order_name not in self.possible_actions.keys() or resource_name not in self.possible_actions[order_name]:
            return "INVALID ACTION, Action is not possible"

        if order.resource:
            self.time = max(resource.free_at, order.resource.free_at, self.time)
        else:
            self.time = max(resource.free_at, self.time)
        
        finish_time = self.time + resource.processing_times[(order.product_name, self.products[order.product_name][order.current_stage ][0][0])] + resource.setup_times[(resource.last_product,order.product_name)] 
        resource.is_occupied = True
        resource.last_product = order.product_name
        resource.free_at= finish_time
        order.increase_stage()
        order.resource = resource

        self.df.append(dict(Task=resource_name, Start=self.time, Finish=finish_time, Resource=order.product_name))

        #self.action_list.put((finish_time, order ))

        if order.current_stage == self.NR_STAGES + 1 :
            self.possible_actions.pop(order_name)
            
            self.last_job = max(finish_time,self.last_job)
            print("Order No: " , order.order_name, " order product: ",order.product_name , " is produced at time: ", finish_time)
            print("Duedate was: ", order.due_date, "\n")
            
            if order.due_date >= finish_time:
                self.success += 1
            if not self.possible_actions:
                self.terminate()

        else:
            self.possible_actions[order_name] = self.products[order.product_name][order.current_stage][0][1]
        """
        while not self.action_list.empty():
            deadline, prod = self.action_list.get()
            prod.resource.is_occupied = False
            if deadline > self.time:
                self.action_list.put((deadline,prod))
                break
        """