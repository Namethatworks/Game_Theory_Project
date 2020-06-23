from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

import ast
import networkx as nx


class Node:
    def __init__(self, name):
        self.neighbours = []
        self.payoffs = dict()
        self.name = name

    def set_name(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    def add_neighbour(self, node):
        self.neighbours.append(node)

    def get_neighbours(self):
        return self.neighbours

    def add_payoff(self, order, condition, value):
        ordered = tuple([item for item in sorted(zip(order, condition), key=lambda x: x[0].get_name())])
        self.payoffs[ordered] = value

    def get_payoff(self, order, condition):
        ordered = tuple([item for item in sorted(zip(order, condition), key=lambda x: x[0])])
        return self.payoffs[ordered]


class ExpressionGraph:
    nodes: Dict[str, Node] = dict()
    main_nodes = 0
    internal_nodes = 0

    def __init__(self, expression):
        ast_tree = ast.parse(expression, mode='eval')
        self._eval(ast_tree.body)

    def plot(self):

        G = nx.DiGraph()
        G.add_nodes_from(self.nodes.keys())
        for node in self.nodes.values():
            for neighbour in node.get_neighbours():
                G.add_edge(neighbour.get_name(), node.get_name())

        pos = nx.spring_layout(G)

        all_nodes = self.nodes.keys()
        end_nodes = ["v" + str(self.main_nodes - 1)]
        main_nodes = [x for x in all_nodes if not x.startswith("v") and not x.startswith("w")]
        int_nodes = [x for x in all_nodes if (x.startswith("v") or x.startswith("w")) and not x == end_nodes[0]]

        nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, node_color="orange", node_size=1000)
        nx.draw_networkx_nodes(G, pos, nodelist=int_nodes, node_color="gray", node_size=1000)
        nx.draw_networkx_nodes(G, pos, nodelist=end_nodes, node_color="lightblue", node_size=1000)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True)
        plt.show()

    def create_main_node(self):
        name = "v" + str(self.main_nodes)
        self.main_nodes = self.main_nodes + 1
        node = self.nodes[name] = Node(name)
        return node

    def create_internal_node(self):
        name = "w" + str(self.internal_nodes)
        self.internal_nodes = self.internal_nodes + 1
        node = self.nodes[name] = Node(name)
        return node

    def create_independent_node(self, name: str):
        if name.startswith("v") or name.startswith("w"):
            raise Exception("variable names must not start with v or w")

        if name not in self.nodes:
            self.nodes[name] = Node(name)
        return self.nodes[name]

    def _eval(self, node):
        if isinstance(node, ast.Name):  # variable name
            return self.create_independent_node(node.id)
        if isinstance(node, ast.Num):  # constant number
            return self._num(node.n)
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            return self.plus_minus_multiply(self._eval(node.left), self._eval(node.right))
        #        elif isinstance(node, ast.U):  # <operator> <operand> e.g., -1
        #            return self.operators[type(node.op)](eval_(node.operand))
        else:
            raise TypeError(node)

    def _num(self, value):
        if value < 0:
            raise Exception("negative values are not supported")

        v1 = self.create_main_node()
        w = self.create_internal_node()

        v1.add_neighbour(w)
        w.add_neighbour(v1)

        v1.add_payoff((v1, w), (0, 0), 0)
        v1.add_payoff((v1, w), (0, 1), 1)
        v1.add_payoff((v1, w), (1, 0), 1)
        v1.add_payoff((v1, w), (1, 1), 0)
        w.add_payoff((v1, w), (0, 0), value)
        w.add_payoff((v1, w), (1, 0), value)
        w.add_payoff((v1, w), (0, 0), 0)
        w.add_payoff((v1, w), (0, 0), 0)

        return v1

    def _multiply_constant(self, value, v1):
        if value < 0:
            raise Exception("negative values are not supported")

        print("Multiplying {} by {}".format(v1.get_name(), value))
        v2 = self.create_main_node()
        w = self.create_internal_node()

        v2.add_neighbour(w)
        w.add_neighbour(v2)
        w.add_neighbour(v1)

        return v2

    def plus_minus_multiply(self, v1, v2):
        print("Binary operating on ", v1.get_name(), v2.get_name())
        # all the same for now, just the payoffs differ
        v3 = self.create_main_node()
        w = self.create_internal_node()

        v3.add_neighbour(w)
        w.add_neighbour(v1)
        w.add_neighbour(v2)
        w.add_neighbour(v3)
        return v3


# Initialize expression graph with whatever expression you want
# due to poor design don't use v and w in the expression variables
# supported operations +, -, *, multiplication with constant,
# TODO unary negation (really simple, was too tired to bother)
# TODO force networkx for draw a well aligned graph
# TODO change colors of nodes and nodes for different types / subgraphs
# TODO there are ways to optimize the produced graph using a more extensive usage of the Propositions 2 and 1

expr = ExpressionGraph("a+b+c*2")
expr.plot()
