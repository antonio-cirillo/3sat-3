from networkx.drawing.layout import bipartite_layout
import networkx as nx

from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import re

import matplotlib

matplotlib.use('TkAgg')
PHI_REGEX = re.compile('^(\(-?\w+ v -?\w+ v -?\w+\))( \^ \((-?\w+ v -?\w+ v -?\w+)\))*$')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def generate_random_formula():
    _variables = ['x1', 'x2', 'x3', 'x4', 'x5']
    n_occurrences = [0, 0, 0, 0, 0]
    _phi = ''

    while 1:
        # get the literals appeared less than 3 times
        temp = []
        for j in range(len(_variables)):
            if n_occurrences[j] < 3:
                temp.append(_variables[j])

        if len(temp) == 0:
            break

        # randomly choose 3 literals
        try:
            x1, x2, x3 = np.random.choice(temp, 3, replace=False)
        # if there are not at least 3 literals, add 2 more literals to the formula
        except ValueError:
            n_variables = len(_variables)
            v1 = f'x{n_variables + 1}'
            v2 = f'x{n_variables + 2}'
            _variables.extend([v1, v2])
            n_occurrences.extend([0, 0])
            continue

        n1, n2, n3 = np.random.choice(['', '-'], 3)

        # insert the clause
        _phi += f'({n1}{x1} v {n2}{x2} v {n3}{x3}) ^ '

        # updates the vector of occurrences
        for j in [x1, x2, x3]:
            n_occurrences[_variables.index(j)] += 1

    return _phi[:-3]


def create_graph(formula):
    _edges = pd.DataFrame(columns=['u', 'v'])
    _literals_for_each_clauses = []
    _clauses = formula.split(' ^ ')
    n_occurrences = []
    _variables = []

    for i in range(len(_clauses)):
        # get literals for each clause
        literals = re.findall('-?\w\d*', _clauses[i])
        l1, l2, l3 = literals[0], literals[2], literals[4]

        # get variables for each clause
        x1, x2, x3 = re.findall('\w\d*', f'{l1} {l2} {l3}')

        # add variables in variables array and updated number of occurrences
        # create edge (u, v) where u = xj and v = Ci
        for j in [x1, x2, x3]:
            try:
                index = _variables.index(j)
                n_occurrences[index] += 1
            except ValueError:
                _variables.append(j)
                n_occurrences.append(1)
            _edge = {"u": j, "v": i + 1}
            _edges = pd.concat([_edges, pd.DataFrame([_edge])])

        # append literals array
        _literals_for_each_clauses.append([l1, l2, l3])

    if min(n_occurrences) != 3:
        print('The formula entered is not a 3sat-3 formula.')
        print('The variables must appear exactly 3 times.')
        exit()

    # generate array of clauses node
    item = [i for i in range(1, len(_clauses))]

    # create and plot graph
    _graph = nx.Graph()
    _graph.add_nodes_from(_variables, bipartite=0)
    _graph.add_nodes_from(item, bipartite=2)
    _graph.add_edges_from(tuple(x) for x in _edges.values)

    pos = bipartite_layout(_graph, _variables)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(_graph, pos, with_labels=True, node_size=1000, node_color='r', edge_color='g', arrowsize=10)
    plt.savefig('./bipartite-graph.jpeg')
    plt.close()

    # create csr_matrix
    matrix = np.zeros((len(_variables), len(_clauses)), dtype=int)
    for i, _edge in _edges.iterrows():
        xi = _edge['u']
        cj = _edge['v']
        matrix[_variables.index(xi)][cj - 1] = 1

    return _variables, _clauses, _literals_for_each_clauses, csr_matrix(matrix)


def plot_matching(_variables, _clauses, _edges):
    # generate array of clauses node
    item = [i for i in range(1, len(_clauses))]

    # create and plot graph
    _graph = nx.Graph()
    _graph.add_nodes_from(_variables, bipartite=0)
    _graph.add_nodes_from(item, bipartite=2)
    _graph.add_edges_from(tuple(_edge) for _edge in _edges.values)

    pos = bipartite_layout(_graph, _variables)
    plt.figure(figsize=(10, 10))
    nx.draw_networkx(_graph, pos, with_labels=True, node_size=1000, node_color='r', edge_color='g', arrowsize=10)
    plt.savefig('./perfect-matching.jpeg')
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gen', dest='gen', action='store',
                        default=False, type=str2bool, help='generate phi formula')
    args = parser.parse_args()

    if args.gen:
        phi = generate_random_formula()
        print('This is the generated formula:')
        print(phi)
    else:
        print('Enter the formula 3sat3 using this format.')
        print('Use "-" for the "NOT" operation. For example: NOT A is equal to -A')
        print('Use "v" to separate literals in clause. For example: (A v B v -C).')
        print('Use "^" to separate clauses from each other. For example: (A v B v -C) ^ (A v E v D)')
        phi = input('Enter the formula: ')
        if not re.fullmatch(PHI_REGEX, phi):
            print('The input entered is not a valid formula.')
            exit()

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
        edges = pd.concat([edges, pd.DataFrame([edge])])

        # if u appears in the v negated clause then u = 0, otherwise 1
        if f'-{u}' in literals_for_each_clauses[i]:
            output[variables.index(u)] = 0
        else:
            output[variables.index(u)] = 1
    # plot matching
    plot_matching(variables, clauses, edges)

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
