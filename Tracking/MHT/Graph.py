#!/usr/bin/env python

###################################################################################################
#                                           Graph Class                                           #
#                                                                                                 #
# Class for creating and maintaining graphs that are used in the Maximum Weighted Independent Set #
# implementation of the multiple hypothesis tracker. It contains classes for both a normal graph, #
# as well as a weighted graph.                                                                    #
#                                                                                                 #
# Author: Marco Grond                                                                             #
# Version: 0.1.0                                                                                  #
###################################################################################################

import numpy as np

class Graph:

    def __init__(self, graphDict=None, directed=False):
        """
        Initialize the graph

        graphDict - A optional dictionary describing a graph
        directed - A flag to set whether the graph should be directed or not
        """

        self.DIRECTED = directed
        if graphDict is None:
            self.GRAPH = {}
        else:
            self.GRAPH = graphDict


    def getGraphDict(self):
        """
        Return the dictionary describing the graph
        """

        return self.GRAPH


    def getVertices(self):
        """
        Return the list of all vertices
        """

        return list(self.GRAPH.keys())


    def getVertexEdges(self, vertex):
        """
        Return a list of all vertices that the provided vertex has an edge to. For a directed 
        graph, only edges originating at the provided vertex are counted.

        vertex - The identifier of the vertex. If the vertex does not exist in the graph, no edges 
                 are returned
        """

        if (vertex in self.GRAPH):
            return list(self.GRAPH[vertex])
        return []


    def getVertexDegree(self, vertex):
        """
        Get the degree for the vertex, that is the number of edges originating from that vertex.

        vertex - The identifier of the vertex. If the vertex does not exist, -1 is returned.
        """

        print(self.GRAPH)
        if (vertex in self.GRAPH):
            return len(self.GRAPH[vertex])
        return -1


    def getAllEdges(self):
        """
        Returns all the edges in the graph as a list of tuples. If the graph is undirected, only 
        a single directional edge is provided, otherwise the first entry of the tuple is the 
        originating vertex for the edge.
        """

        edgeList = set()

        # For a directed graph, add each edge
        if self.DIRECTED:
            for vertex1 in self.GRAPH:
                for vertex2 in self.GRAPH[vertex1]:
                    edgeList.add((vertex1, vertex2))

        # For an undirected graph, only add a single edge between two vertices
        else:
            vertexList = list(self.GRAPH.keys())
            vertexList.sort()
            for vertex1 in vertexList:
                for vertex2 in self.GRAPH[vertex1]:
                    if (vertex1 <= vertex2):
                        edgeList.add((vertex1, vertex2))

        return list(edgeList)
                


    def addVertex(self, vertexName, edgeList=[]):
        """
        Add a vertex to the graph and optionally add edges for that vertex

        vertexName - The identifier for the vertex
        edgeList - A list of vertices that the new vertex should have edges with. If a vertex in 
                   the list does not exist, it is simply ignored
        """

        # Check to see that an existing vertex is not being overwritten
        if (vertexName in self.GRAPH):
            print("Vertex", vertexName, "is already in the graph")
            return

        # Create a vertex and set the edges for it
        self.GRAPH[vertexName] = set()
        for edge in edgeList:
            if (edge in self.GRAPH):
                self.GRAPH[vertexName].add(edge)
                if not self.DIRECTED:
                    self.GRAPH[edge].add(vertexName)


    def addEdge(self, edge):
        """
        Add the given edge, which is of the type set, tuple or list, to the graph, if both vertices
        exist. If either vertex does not exist, the edge is not added. For directed graphs, a list
        or tuple should be used where the first entry is the originating vertex for the edge.

        edge - A set, tuple or list of two vertices between which an edge should be created.
        """

        vertex1, vertex2 = tuple(edge)
        if ((vertex1 in self.GRAPH) and (vertex2 in self.GRAPH)):
            self.GRAPH[vertex1].add(vertex2)
            if not self.DIRECTED:
                self.GRAPH[vertex2].add(vertex1)


    def addMultipleEdges(self, edgeList):
        """
        Add multiple edges to the graph. Each edge should be a set, tuple or list containing two 
        vertices between which an edge should be created. For directed graphs, an edge should be a
        tuple or list where the first entry is the originating vertex for the edge. Edges are only
        added if a vertex exists.

        edgeList - A list of edges
        """

        for edge in edgeList:
            self.addEdge(edge)


    def removeVertex(self, vertexName):
        """
        Removes the given vertex and all edges associated with it

        vertexName - The identifier of the vertex that should be removed
        """

        if (vertexName in self.GRAPH):

            # If it is a directed graph, need to remove all other edges that contain the vertex
            if self.DIRECTED:
                self.GRAPH.pop(vertexName)
                for vertex in self.GRAPH:
                    self.GRAPH[vertex].discard(vertexName)

            # If it is an undirected graph, need to only remove edges that originate at the removed
            # vertex
            else:
                edgeList = self.GRAPH.pop(vertexName)
                for vertex in edgeList:
                    self.GRAPH[vertex].discard(vertexName)


    def removeEdge(self, edge):
        """
        Removes a single edge from the graph, given as two vertices as a set, list or tuple. If a 
        directed graph is used, the edge should be a list or tuple, where the first entry is the 
        originating vertex for the edge.

        edge - A set, list or tuple of two vertices denoting the edge that should be removed
        """

        (vertex1, vertex2) = tuple(edge)
        if ((vertex1 in self.GRAPH) and (vertex2 in self.GRAPH)):
            if self.DIRECTED:
                self.GRAPH[vertex1].discard(vertex2)
            else:
                self.GRAPH[vertex1].discard(vertex2)
                self.GRAPH[vertex2].discard(vertex1)


    def removeMultipleEdges(self, edgeList):
        """
        Remove multiple edges from the graph, where each edge is given as two vertices as a set, 
        list or a tuple. If a directed graph is used, an edge should be a list or a tuple, where 
        the first entry is the originating vertex for the edge.

        edgeList - The list of edges to be removed
        """

        for edge in edgeList:
            self.removeEdge(edge)


    def getAdjacencyMatrix(self):
        """
        Compute the adjacency matrix for the graph. Returns the adjacency matrix as a numpy array
        as well as a list of labels for each row/column of the matrix.

        returns - AdjacencyMatrix, LabelsList
        """

        # Initialize the adjacency matrix and the labels for it
        matrixSize = len(self.GRAPH.keys())
        adjMatrix = np.zeros((matrixSize, matrixSize))
        labelList = list(self.GRAPH.keys())
        labelList.sort()

        # Add each edge to the adjacency matrix
        for i, vertex1 in enumerate(labelList):
            for vertex2 in self.GRAPH[vertex1]:
                j = labelList.index(vertex2)
                adjMatrix[i, j] = 1

        return adjMatrix, labelList


    def getComplement(self):
        """
        Compute the adjacency matrix for the complement of the graph. Returns the adjacency matrix
        as a numpy array as well as a list of labels for each row/column of the matrix.

        returns - AdjacencyMatrix, LabelsList
        """

        adjMatrix, labelList = self.getAdjacencyMatrix()
        fullGraph = np.ones(adjMatrix.shape)
        np.fill_diagonal(fullGraph, 0)
        compMatrix = fullGraph - adjMatrix
        return compMatrix, labelList


    def getComplementGraph(self):
        """
        Compute the complement graph of this graph as a dictionary.
        """

        complement = {}
        allKeys = set(self.GRAPH.keys())
        for key in allKeys:
            complement[key] = set.difference(allKeys, self.GRAPH[key], {key})
        return complement
