import numpy as np
import json
from queue import PriorityQueue
import random, os
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

    def __init__(self, visualise, verbose):

        self.DATA_PATH = "data/"

        self.initialise_data()

        # Accessing data

        self.visualise = visualise
        self.verbose = verbose
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
        self.orders_not_initialise = {order_name:Order(order_name, args["due_date"], args["product"]) for order_name, args in self.data["orders"].items()}

        self.action_list = PriorityQueue()
        # [(finish_time, order, resource) ... ]

        self.waiting_action_list = []

        self.time = 0
        self.success = 0
        self.last_job = 0
        self.alive = True

        self.possible_actions = {order.order_name: self.products[order.product_name][order.current_stage][0][1].copy() for order in self.orders_not_initialise.values()}


    def initialise_data(self):

        files = os.listdir(self.DATA_PATH)
        selected_file = random.choice(files)

        with open(os.path.join(self.DATA_PATH, selected_file), 'r') as file:
            self.data = json.load(file)
            

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
        
        finish_time = self.time + resource.processing_times[(order.product_name, self.products[order.product_name][order.current_stage ][0][0])] + resource.setup_times[(resource.last_product,order.product_name)] 
        resource.is_occupied = True
        resource.last_product = order.product_name
        resource.free_at= finish_time

        if order.current_stage == 1:
            self.orders_not_initialise.pop(order_name)

        for i, temp_ord in enumerate(self.waiting_action_list):
            if temp_ord.order_name == order_name:
                self.waiting_action_list.pop(i)
                break

        order.increase_stage()
        order.resource = resource_name

        self.df.append(dict(Task=resource_name, Start=self.time, Finish=finish_time, Resource=order.product_name))

        self.action_list.put((finish_time, order ))

        self.possible_actions.pop(order_name)
        deleted = []
        for o,rl in self.possible_actions.items():
            if resource_name in rl:
                rl.pop(rl.index(resource_name))

            if not rl:
                deleted.append(o)

        for o in deleted:
            self.possible_actions.pop(o)

        while not self.possible_actions:
            self.time, ord_t = self.action_list.get()

            finished_product = False

            if ord_t.current_stage == self.NR_STAGES + 1 :

                finished_product = True
                
                if self.verbose:
                    print("Order No: " , order.order_name, " order product: ",order.product_name , " is produced at time: ", finish_time)
                    print("Duedate was: ", order.due_date, "\n")
                
                if order.due_date >= finish_time:
                    self.success += 1

                if not self.possible_actions and self.action_list.empty() and not self.waiting_action_list:
                    self.terminate()
                    break
                
                
            else:
                resources = self.products[ord_t.product_name][ord_t.current_stage][0][1]
                filtered_resorces = []
                for res in resources:
                    if not self.resources[res].is_occupied:
                        filtered_resorces.append(res)

                if filtered_resorces:
                    self.possible_actions[ord_t.order_name] = filtered_resorces

            free_resource_name = ord_t.resource
            ord_t.resource = None
            free_resource = self.resources[free_resource_name]
            free_resource.is_occupied = False

            if ord_t.current_stage == 2:
                #print(self.products)
                for ord_tt in self.orders_not_initialise.values():
                    resources = self.products[ord_tt.product_name][ord_tt.current_stage][0][1]
                    #print(ord_tt.order_name,resources,ord_tt.product_name,ord_tt.current_stage)
                    if free_resource_name in resources:
                        
                        try:
                            self.possible_actions[ord_tt.order_name].append(free_resource_name)
                        except KeyError:
                            self.possible_actions[ord_tt.order_name] = [free_resource_name]


            else:
                for ord_tt in self.waiting_action_list:
                    resources = self.products[ord_tt.product_name][ord_tt.current_stage][0][1]
                    if free_resource_name in resources:
                        try:
                            self.possible_actions[ord_tt.order_name].append(free_resource_name)
                        except KeyError:
                            self.possible_actions[ord_tt.order_name] = [free_resource_name]

            if not finished_product:
                self.waiting_action_list.append(ord_t)
