#!/usr/bin/env python

###################################################################################################
#                                         Assignment Code                                         #
#                                                                                                 #
# Class for applying Murty's algorithm to generate the k-best assignment problem solutions from   #
# best to worst.                                                                                  #
#                                                                                                 #
# Author: Marco Grond                                                                             #
# Version: 0.1.0                                                                                  #
###################################################################################################

import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
import time

class assignment:

    def __init__(self, graphList, rowLabels, columnLabels, maximize=True, debug=False):
        """
        Initialize the queue with the given matrices as well as their matrixlabels

        matrixList - A list of assignment matrices
        rowLabels - A list of label assignments for each assignment matrix where each value is the 
                    label for the corresponding row-index in the corresponding assignment matrix
        columnLabels - A list of label assignments for each assignment matrix where each value is 
                       the label for the corresponding column-index in the corresponding assignment 
                       matrix
        maximize - A flag for whether or not the assignment problem should be maximized (True) or 
                   minimized (False)
        """

        # Set up some initialization variables
        self.ASSIGNMENT_QUEUE = []
        self.DEBUG = debug
        self.MAXIMIZE = maximize
        if self.MAXIMIZE:
            self.MAXIMIZE_MULT = -1
            self.REMOVE_VAL = -np.inf
        else:
            self.MAXIMIZE_MULT = 1
            self.REMOVE_VAL = np.inf
        self.COUNTER = 0

        # Create the assignment priority queue for the given initial graphs
        for index, graph in enumerate(graphList):
            isValid, matrix, lookup = graph.computeAssignmentMatrix(self.REMOVE_VAL)
            if isValid:
                rows, cols = hungarian(matrix, maximize=self.MAXIMIZE)
                choices = [lookup[r][c] for r, c in zip(rows, cols)]
                solutionCost, solution = graph.getSolution(choices)
                heapq.heappush(self.ASSIGNMENT_QUEUE, (self.MAXIMIZE_MULT*solutionCost, 
                                                       self.COUNTER, solution, graph, index))
            self.COUNTER += 1


    def getDefaultValue(self):
        """
        Returns the default value used to indicate that an assignment in the matrix is not allowed
        """

        return self.REMOVE_VAL


    def getBest(self, minAllowed):
        """
        Remove and return the best assignment in the priority queue and add new assignment problems 
        using Murty's algorithm

        Returns - A tuple in the form (cost, assignment) where cost is the total cost for the given 
                  assignment, and assignment is a list of assignments, where each item in the list 
                  is a tuple where the first index is the row label and the second index is the 
                  column label of the solution.
        """

        counter = 0

        if (len(self.ASSIGNMENT_QUEUE) == 0):
            return (self.REMOVE_VAL, [], -1)

        # Get the best solution from the priority queue
        bestCost, _, bestSolution, bestGraph, index = heapq.heappop(self.ASSIGNMENT_QUEUE)
        bestCost *= self.MAXIMIZE_MULT
        if (bestCost < minAllowed):
            return (self.REMOVE_VAL, [], -1)
        if self.DEBUG:
            print("#"*80)
            print("Getting Best Solution To Return.")
            print("Best Solution Cost:", bestCost)
            print("Best Solution:", bestSolution)
            print("#"*80)
            sortedSequence = heapq.nsmallest(len(self.ASSIGNMENT_QUEUE), self.ASSIGNMENT_QUEUE)
            print("Total number of possible solutions:", len(self.ASSIGNMENT_QUEUE))
            print("#"*80)
            for a, b, c, d, e in sortedSequence:
                print("Solution Cost:", self.MAXIMIZE_MULT*a)
                print("Solution:", c)
                print("-"*80)
            print("#"*80)

        # Use Murty's algorithm to add new possible solutions to the priority queue
        if (not bestGraph.isEmpty()):

            # Add new possible solutions to the heap
            for node in bestSolution:
                if (node is None):
                    continue

                # Remove the node from the graph
                if (bestGraph.hasNode(node.getNodeID())):
                    newGraph = bestGraph.removeNode(node.getNodeID())

                    # Check to see if the number of available solutions is still valid
                    isValid, matrix, lookup = newGraph.computeAssignmentMatrix(self.REMOVE_VAL)
                    if not isValid:
                        continue

                    # Perform data association using the hungarian algorithm
                    try:
                        rows, cols = hungarian(matrix, maximize=self.MAXIMIZE)
                    except:
                        continue

                    # Compute the cost for this association choice
                    choices = [lookup[r][c] for r, c in zip(rows, cols)]
                    solutionCost, solution = newGraph.getSolution(choices)

                    if self.DEBUG:
                        print("Adding to priority queue...")
                        print("Removed: (", str(node.getNodeID()) + " - (" + str(node.getTreeID()) + ", " + str(node.getDetectID()) + "))")
                        print("Solution Cost:", solutionCost)
                        print("Solution:", solution)
                        print("Matrix:")
                        print(matrix)
                        print("Lookup:", lookup)
                        print()

                    heapq.heappush(self.ASSIGNMENT_QUEUE, (self.MAXIMIZE_MULT*solutionCost, self.COUNTER, solution, newGraph, index)) 
                    self.COUNTER += 1

                    # Make a hard decision on (r, c), remove it from the assignment matrix
                    bestGraph = bestGraph.makeHardChoice(node.getNodeID())

        return (bestCost, bestSolution, index)


    def hasSolution(self):
        """
        Returns whether or not there are potential solutions that can be computed.
        """

        return (len(self.ASSIGNMENT_QUEUE) != 0)


    def clearQueue(self):
        """
        Empties the current queue
        """

        self.ASSIGNMENT_QUEUE = []


    def setQueue(self, graphList):
        """
        Clear and set the assignment queue using the list of provided graph-hypotheses

        graphList - A list of graph hypotheses, that have already been filled
        """

        self.ASSIGNMENT_QUEUE = []
        self.COUNTER = 0

        # Create the assignment priority queue for the given initial graphs
        for index, graph in enumerate(graphList):
            isValid, matrix, lookup = graph.computeAssignmentMatrix(self.REMOVE_VAL)
            if isValid:
                rows, cols = hungarian(matrix, maximize=self.MAXIMIZE)
                choices = [lookup[r][c] for r, c in zip(rows, cols)]
                solutionCost, solution = graph.getSolution(choices)
                heapq.heappush(self.ASSIGNMENT_QUEUE, (self.MAXIMIZE_MULT*solutionCost, 
                                                       self.COUNTER, solution, graph, index))
            self.COUNTER += 1



if __name__ == '__main__':
    a = np.array([[82, 83, 69, 92], [77, 37, 49, 92], [11, 69, 5, 86], [8, 9, 98, 23]])
    #a = np.random.rand(7, 7)
    #b = np.random.rand(7, 7)
    maximize = True
    numPrint = 100
    #start = time.time()
    #ranked1 = assignMurty(a, numPrint, maximize, False)
    #ranked2 = assignMurty(b, numPrint, maximize, False)
    #end = time.time()
    #ranked = []
    #for c1, c2 in ranked1:
    #    heapq.heappush(ranked, (-c1, c2))
    #for c1, c2 in ranked2:
    #    heapq.heappush(ranked, (-c1, c2))
    #final = []
    #for i in range(numPrint):
    #    c1, c2 = heapq.heappop(ranked)
    #    final.append((-c1, c2))
    #print("Number of assignments:", len(final))
    #for cost, choices in final:
    #    print("Cost:", cost)
    #    print("Choices:", choices)
    #print("Total time:", (end - start))
    #print()

    assign = assignment([a], [np.arange(1, 5)], [np.arange(1, 5)], maximize, True)
    #assign = assignment([a, b], [np.arange(6, -1, -1), np.arange(6, -1, -1)], [np.arange(6, -1, -1), np.arange(6, -1, -1)], maximize, True)
    costs = []
    choices = []
    start = time.time()
    for i in range(numPrint):
        c1, c2, index = assign.getBest()
        costs.append(c1)
        choices.append(c2)
    end = time.time()
    print("Number of assignments:", len(costs))
    for cost, choices in zip(costs, choices):
        print("Cost:", cost)
        print("Choices:", choices)
    print("Total time:", (end - start))
        

    #print()
    #start = time.time()
    #out = bruteForce(a, numPrint, maximize)
    #end = time.time()
    #for cost, choices in out:
    #    print("Cost:", cost, )
    #    print("Choices:", choices)
    #print("Total time:", (end - start))
