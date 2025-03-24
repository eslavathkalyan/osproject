import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import networkx as nx

class AIDEDeadlockEngine:
    def __init__(self):
        self.resource_allocation_graph = nx.DiGraph()
        self.prediction_model = None
        self.historical_data = pd.DataFrame()

    def collect_data(self, process_id, resource_id, request_type):
        if request_type == "request":
            self.resource_allocation_graph.add_edge(process_id, resource_id)
        elif request_type == "allocate":
            self.resource_allocation_graph.add_edge(resource_id, process_id)
        elif request_type == "release":
            self.resource_allocation_graph.remove_edge(resource_id, process_id)

    def train_prediction_model(self, data):
        X = data.drop("deadlock", axis=1)
        y = data["deadlock"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.prediction_model = LogisticRegression()
        self.prediction_model.fit(X_train, y_train)

    def predict_deadlock(self, current_state_features):
        if self.prediction_model:
            prediction = self.prediction_model.predict(current_state_features)
            return prediction[0]
        else:
            return False

    def detect_deadlock(self):
        try:
            cycle = nx.find_cycle(self.resource_allocation_graph)
            return True, cycle
        except nx.NetworkXNoCycle:
            return False, None

    def resolve_deadlock(self):
        if self.detect_deadlock()[0]:
            cycle = self.detect_deadlock()[1]
            process_to_terminate = cycle[0][0]
            print(f"Deadlock detected. Terminating process: {process_to_terminate}")
            self.resource_allocation_graph.remove_node(process_to_terminate)
            return True
        else:
            return False
            
    def get_graph(self):
        return self.resource_allocation_graph