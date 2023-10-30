#!/usr/bin/env python

###################################################################################################
#                                       Weighted Graph Class                                      #
#                                                                                                 #
# Class that extends the Graph class to allow for weights to be associated with the vertices, and #
# enables the graph to be used with a Multiple Hypothesis Tracker.                                #
#                                                                                                 #
# Author: Marco Grond                                                                             #
# Version: 0.1.0                                                                                  #
###################################################################################################

import numpy as np
from Graph import Graph
import random

import time

class WeightedGraph(Graph):

    def __init__(self, graphDict=None, weights=None, debug=False):
        """
        Initialize the graph. In order to use many of the methods, the graph must be undirected
        and so that is forced.

        graphDict - A optional dictionary describing a graph
        weights - An optional dictionary denoting the weight of each vertex. All missing vertices
                  are initialized with a weight of zero

        """

        Graph.__init__(self, graphDict, False)
        self.WEIGHTS = {}
        if not ((graphDict is None) or (weights is None)):
            for key in graphDict.keys():
                if key in weights:
                    self.WEIGHTS[key] = weights[key]
                else:
                    self.WEIGHTS[key] = 0

        self.DEBUG = debug


    def getMWIS(self):
        """
        Determine the maximum weighted independent set for the graph. If a node does not have a 
        weight associated with it, a zero weight is used.
        """

        # Find all maximal cliques for the complement of the graph (These would be independent 
        #sets for the normal graph)
        comp = self.getComplementGraph()
        independentSets = []
        self.bronKerbosch3(comp, independentSets)

        # Find the maximum weighted independent set
        if (len(independentSets) == 0):
            return []
        mwis = independentSets[0]
        maxWeight = sum([self.WEIGHTS[v] for v in mwis])
        for indSet in independentSets[1:]:
            weight = sum([self.WEIGHTS[v] for v in indSet])
            if (weight > maxWeight):
                maxWeight = weight
                mwis = indSet

        return mwis


    def addWeigthedVertex(self, vertexName, vertexWeight, edgeList=[]):
        """
        Add a single vertex with the given weight to the graph.

        vertexName - The identifier used for the vertex
        vertexWeight - The weight associated with the vertex
        edgeList - A list of vertices that the new vertex should have edges with. If a vertex in 
                   the list does not exist, it is simply ignored
        """

        self.addVertex(vertexName, edgeList)
        self.WEIGHTS[vertexName] = vertexWeight


    def setVertexWeight(self, vertexName, vertexWeight):
        """
        Update the weight for an existing vertex. If the vertex does not exist, nothing is done.

        vertexName - The identifier of the vertex whose weight should be updated
        vertexWeight - The weight that should be assigned to the vertex
        """

        if (vertexName in self.WEIGHTS):
            self.WEIGHTS[vertexName] = vertexWeight


    def bronKerbosch(self, clique, candidates, excluded, neighbours, results):
        """
        Implementation of the recursive Bron-Kerbosch algorithm to find all maximal cliques in an 
        undirected graph.

        cliques - Independent vertex cliques that have been formed up to this point. Should be 
                  empty for the first call of the algorithm.
        candidates - A set of vertices that may be added to the clique for the next iteration of
                     the algorithm
        excluded - A set of vertices that should be excluded from being added to the next iteration
                   of the algorithm. Should be empty for the first call of the algorithm.
        neighbours - A dictionary of a set of the neighbours for each vertex
        results - A list in which the final clique results can be stored
        """

        # Check to see if the lists of candidates and excluded vertices are empty, in which case 
        # the current clique is maximal
        if not any(set.union(candidates, excluded)):
            results.append(clique)

        # Perform the Bron-Kerbosch recursion
        for vertex in candidates:
            vertexNeighbours = neighbours[vertex]
            self.bronKerbosch(clique.union({vertex}), candidates.intersection(vertexNeighbours), 
                              excluded.intersection(vertexNeighbours), neighbours, results)
            candidates = candidates.difference({vertex})
            excluded.add(vertex)


    def bronKerbosch2(self, clique, candidates, excluded, neighbours, results):
        """
        Implementation of the Bron-Kerbosch algorithm, with pivoting, to find all maximal cliques
        in an undirected graph.

        cliques - Independent vertex cliques that have been formed up to this point. Should be 
                  empty for the first call of the algorithm.
        candidates - A set of vertices that may be added to the clique for the next iteration of
                     the algorithm
        excluded - A set of vertices that should be excluded from being added to the next iteration
                   of the algorithm. Should be empty for the first call of the algorithm.
        neighbours - A dictionary of a set of the neighbours for each vertex
        results - A list in which the final clique results can be stored
        """

        # Check to see if the lists of candidates and excluded vertices are empty, in which case 
        # the current clique is maximal
        candidates_excluded = candidates.union(excluded)
        #print('\t', candidates, excluded, candidates_excluded)
        if (len(candidates_excluded) == 0):
            results.append(clique)
            return

        # Choose a random pivot vertex from the union of the candidates and excluded items
        pivot = random.choice(tuple(candidates_excluded))
        #print('\t', 'Pivot', pivot, neighbours[pivot])

        # Only check if the pivot vertex or one of its non-neighbours should belong to the clique
        for vertex in candidates.difference(neighbours[pivot]):
            vertexNeighbours = neighbours[vertex]
            self.bronKerbosch2(clique.union({vertex}), candidates.intersection(vertexNeighbours), 
                              excluded.intersection(vertexNeighbours), neighbours, results)
            candidates = candidates.difference({vertex})
            excluded.add(vertex)


    def bronKerbosch3(self, neighbours, results):
        """
        Implementation of the Bron-Kerbosch algorithm, with degeneracy ordering, to find all 
        maximal cliques in an undirected graph.

        neighbours - A dictionary of a set of the neighbours for each vertex
        results - A list in which the final clique results can be stored
        """

        # Set the candidates to be all vertices and compute the degeneracy ordering
        candidates = set(neighbours.keys())
        cliques, excluded = set(), set()
        degenOrder = self.degeneracyOrdering(neighbours)

        # Perform Bron-Kerbosch with pivoting on vertices based on their degeneracy ordering
        for vertex in degenOrder:
            vertexNeighbours = neighbours[vertex]
            #print("Candidates, Excluded, Cliques, Results, Vertex, VertexNeighbours")
            #print(candidates, excluded, (cliques | {vertex}), results, vertex, vertexNeighbours)
            self.bronKerbosch2(cliques | {vertex}, candidates.intersection(vertexNeighbours), 
                              excluded.intersection(vertexNeighbours), neighbours, results)
            candidates = candidates.difference({vertex})
            excluded.add(vertex)


    def degeneracyOrdering(self, neighbours):
        """
        Perform a degeneracy ordering, which returns a list of all of the vertices provided sorted
        in decending order by the number of neighbours that a vertex has

        neighbours - A dictionary of a set of the neighbours for each vertex
        """

        # Compute the number of neighbours for each vertex
        vertices = list(neighbours.keys())
        degrees = [len(neighbours[v]) for v in vertices]

        # Sort the vertices in decending number of neighbours
        ordering = [v for d, v in sorted(zip(degrees, vertices), reverse=True)]
        return ordering


if __name__ == '__main__':
    '''
    neighbours = {1:{2, 3, 4},
                  2:{1, 3, 4, 5},
                  3:{1, 2, 4, 5},
                  4:{1, 2, 3},
                  5:{2, 3, 6, 7},
                  6:{5, 7},
                  7:{5, 6}}
    neighbourMat = np.array([[0, 1, 1, 1, 0, 0, 0],
                             [1, 0, 1, 1, 1, 0, 0],
                             [1, 1, 0, 1, 1, 0, 0],
                             [1, 1, 1, 0, 0, 0, 0],
                             [0, 1, 1, 0, 0, 1, 1],
                             [0, 0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 0, 1, 1, 0]])
    wg = WeightedGraph(neighbours)
    startTime = time.time()
    for i in range(1000):
        results = []
        candidates = {1, 2, 3, 4, 5, 6, 7}
        excluded = set()
        cliques = set()
        wg.bronKerbosch(cliques, candidates, excluded, neighbours, results)
    endTime = time.time()
    print("Bron Kerbosch 1:", ((endTime - startTime)/1000))
    print(results)
    print()

    startTime = time.time()
    for i in range(1000):
        results = []
        candidates = {1, 2, 3, 4, 5, 6, 7}
        excluded = set()
        cliques = set()
        wg.bronKerbosch2(cliques, candidates, excluded, neighbours, results)
    endTime = time.time()
    print("Bron Kerbosch 2:", ((endTime - startTime)/1000))
    print(results)
    print()

    startTime = time.time()
    for i in range(1000):
        results = []
        wg.bronKerbosch3(neighbours, results)
    endTime = time.time()
    print("Bron Kerbosch 3:", ((endTime - startTime)/1000))
    print(results)
    print()
    '''

    neighbours = {1:{2, 3, 4},
                  2:{1, 3, 4},
                  3:{1, 2, 4, 5, 6},
                  4:{1, 2, 3, 5, 6},
                  5:{3, 4, 6, 7},
                  6:{3, 4, 5, 7},
                  7:{5, 6},
                  8:{}}
    g = Graph(neighbours)
    comp = g.getComplementGraph()
    wg = WeightedGraph(comp)
    #print(comp)
    #results = []
    #candidates = {1, 2, 3, 4, 5, 6, 7, 8}
    #excluded = set()
    #cliques = set()
    #wg.bronKerbosch(cliques, candidates, excluded, comp, results)
    #print(results)
    #results = []
    #candidates = {1, 2, 3, 4, 5, 6, 7, 8}
    #excluded = set()
    #cliques = set()
    #wg.bronKerbosch2(cliques, candidates, excluded, comp, results)
    #print(results)
    startTime = time.time()
    for i in range(10000):
        results = []
        wg.bronKerbosch3(comp, results)
    endTime = time.time()
    print(results)
    print((endTime - startTime)/10000)
