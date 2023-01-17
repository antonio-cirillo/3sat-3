from networkx.drawing.layout import bipartite_layout
import networkx as nx

from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

import matplotlib

matplotlib.use('TkAgg')
PHI_REGEX = re.compile('^(\(-?\w+ v -?\w+ v -?\w+\))( \^ \((-?\w+ v -?\w+ v -?\w+)\))*$')


def generate_random_formula():
    # variables = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    variables = ['A', 'B', 'C', 'D']
    n_occurrences = [0, 0, 0, 0, 0, 0, 0, 0]
    _phi = ''

    while 1:
        # get the literals appeared less than 3 times
        temp = []
        for j in range(len(variables)):
            if n_occurrences[j] < 3:
                temp.append(variables[j])

        if len(temp) == 0:
            break

        # randomly choose 3 literals
        try:
            x1, x2, x3 = np.random.choice(temp, 3, replace=False)
        # if there are not at least 3 literals, add 2 more literals to the formula
        except ValueError:
            max_variables = max(variables)
            v1 = chr(ord(max_variables) + 1)
            v2 = chr(ord(max_variables) + 2)
            variables.extend([v1, v2])
            n_occurrences.extend([0, 0])
            continue

        n1, n2, n3 = np.random.choice(['', '-'], 3)

        # insert the clause
        _phi += f'({n1}{x1} v {n2}{x2} v {n3}{x3}) ^ '

        # updates the vector of occurrences
        for j in [x1, x2, x3]:
            n_occurrences[variables.index(j)] += 1

    return _phi[:-3]


def create_graph(formula):
    edges = pd.DataFrame(columns=['u', 'v'])
    literals_for_each_clauses = []
    clauses = formula.split(' ^ ')
    n_occurrences = []
    variables = []

    for i in range(len(clauses)):
        # get literals for each clause
        literals = re.findall('-?\w', clauses[i])
        l1, l2, l3 = literals[0], literals[2], literals[4]

        # get variables for each clause
        x1, x2, x3 = re.findall('\w', f'{l1} {l2} {l3}')

        # add variables in variables array and updated number of occurrences
        # create edge (u, v) where u = xj and v = Ci
        for j in [x1, x2, x3]:
            try:
                index = variables.index(j)
                n_occurrences[index] += 1
            except ValueError:
                variables.append(j)
                n_occurrences.append(1)
            edge = {"u": j, "v": i + 1}
            edges = pd.concat([edges, pd.DataFrame([edge])])

        # append literals array
        literals_for_each_clauses.append([l1, l2, l3])

    if min(n_occurrences) != 3:
        print('The formula entered is not a 3sat-3 formula.')
        print('The variables must appear exactly 3 times.')
        exit()

    # generate array of clauses node
    item = [i for i in range(1, len(clauses))]

    # create and plot graph
    graph = nx.Graph()
    graph.add_nodes_from(variables, bipartite=0)
    graph.add_nodes_from(item, bipartite=2)
    graph.add_edges_from(tuple(x) for x in edges.values)

    pos = bipartite_layout(graph, variables)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(graph, pos, with_labels=True, node_size=1000, node_color='r', edge_color='g', arrowsize=10)
    plt.savefig('./bipartite-graph.jpeg')
    plt.close()

    # create csr_matrix
    matrix = np.zeros((len(variables), len(clauses)), dtype=int)
    for i, edge in edges.iterrows():
        xi = edge['u']
        cj = edge['v']
        matrix[variables.index(xi)][cj - 1] = 1

    return variables, clauses, literals_for_each_clauses, csr_matrix(matrix)


def plot_matching(variables, clauses, edges):
    # generate array of clauses node
    item = [i for i in range(1, len(clauses))]

    # create and plot graph
    graph = nx.Graph()
    graph.add_nodes_from(variables, bipartite=0)
    graph.add_nodes_from(item, bipartite=2)
    graph.add_edges_from(tuple(x) for x in edges.values)

    pos = bipartite_layout(graph, variables)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(graph, pos, with_labels=True, node_size=1000, node_color='r', edge_color='g', arrowsize=10)
    plt.savefig('./perfect-matching.jpeg')
    plt.close()


if __name__ == '__main__':
    """
    print('Enter the formula 3sat3 using this format.')
    print('Use "-" for the "NOT" operation. For example: NOT A is equal to -A')
    print('Use "v" to separate literals in clause. For example: (A v B v -C).')
    print('Use "^" to separate clauses from each other. For example: (A v B v -C) ^ (A v E v D)')
    formula = input('Enter the formula: ')
    """
    phi = generate_random_formula()
    print('This is the generated formula:')
    print(phi)

    # create graph
    variables, clauses, literals_for_each_clauses, graph = create_graph(phi)
    # get the maximum bipartite matching
    matching = maximum_bipartite_matching(graph, perm_type='row')

    # init output
    output = [0] * len(variables)

    # get edges of matching and get assignment value of variables
    edges = pd.DataFrame(columns=['u', 'v'])
    for i in range(len(clauses)):
        u = variables[matching[i]]
        v = i + 1
        edge = {"u": u, "v": v}
        print(edge)
        edges = pd.concat([edges, pd.DataFrame([edge])])

        # if u appears in the v negated clause then u = 0, otherwise 1
        if f'-{u}' in literals_for_each_clauses[i]:
            output[i] = 0
        else:
            output[i] = 1
    # plot matching
    plot_matching(variables, clauses, edges)

    # print assignment
    print('The assignment that satisfies the formula is as follows:')
    for i in range(len(variables)):
        print(f'{variables[i]}: {"False" if output[i] == 0 else "True"}')

    # print the satisfied formula
    output_phi = ''
    for literals_clause in literals_for_each_clauses:
        x = [0, 0, 0]
        for i in range(3):
            if '-' in literals_clause[i]:
                xi = literals_clause[i][1:]
                negative = True
            else:
                xi = literals_clause[i]
                negative = False
            x[i] = output[variables.index(xi)]
            if negative:
                x[i] = int(not x[i])
        x1, x2, x3 = x
        output_phi += f'({x1} v {x2} v {x3}) ^ '
    output_phi = output_phi[:-3]
    print('Having assigned values to the variables, the resulting formula is as follows:')
    print(output_phi)