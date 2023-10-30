#!/usr/bin/env python

###################################################################################################
#                                          Hypothesis Class                                       #
#                                                                                                 #
# Class for creating and maintaining hypotheses that are used for multiple hypothesis tracking. A #
# hypothesis consists of multiple nodes that are independent when it comes to detections, but     #
# whose leaves might have possible clashes. This class allows for the adding of nodes to a        #
# hypothesis as well as removing nodes, making hard decisions on a node and generating            #
# association matrices for a hypothesis.                                                          #
#                                                                                                 #
# Author: Marco Grond                                                                             #
# Version: 0.1.0                                                                                  #
###################################################################################################

import heapq
import numpy as np
from copy import deepcopy
from Tree import Tree

class HypoGraph:

    def __init__(self, populatedHypo=None, hardDecisions=None, trackSet=None, detectionSet=None,
                 graphWeight=0):
        """
        Creates a new instance of the Hypothesis Graph class, where a graph is created to represent
        all possible associations that can be made for a single hypothesis

        populatedGraph - An optional argument that contains an already created graph
        hardDecisions - A list of hard decision that have previously been made for this graph
        trackSet - A set contianing all of the individual track numbers
        detectionSet - A set containing all of the individual detection numbers
        hardDetects - The number of hard decisions that correspond to an actual detection
        graphWeight - The combined weight/cost of the hard decision that have been made
        """

        # Initialize the graph, list of hard decisions and sets for all tracks and detections
        self.HYPO = populatedHypo
        if (populatedHypo is None):
            self.HYPO = {}
        self.HARDDECISIONS = hardDecisions
        if (hardDecisions is None):
            self.HARDDECISIONS = []
        self.ALLTRACKS = trackSet
        if (trackSet == None):
            self.ALLTRACKS = set()
        self.ALLDETECTS = detectionSet
        if (detectionSet == None):
            self.ALLDETECTS = set()

        # Initialize other integer variables
        self.CURRENTWEIGHT = graphWeight


    def addNode(self, node):
        """
        Add a node to the hypothesis

        node - The node containing the track number, detection number and the weight of the 
               detection being assigned to the specific track
        """

        # Get information about the node
        trackNum = node.getTreeID()
        detectNum = node.getDetectID()
        nodeID = node.getNodeID()

        # Add the node to the list of options for this hypothesis
        self.HYPO[nodeID] = node
        self.ALLTRACKS.add(trackNum)
        if (detectNum > 0):
            self.ALLDETECTS.add(detectNum)


    def computeAssignmentMatrix(self, defaultValue):
        """
        Computes an assignment matrix for the current hypothesis. The rows of the assignment matrix
        correspond to unique tracks while the columns correspond to possible detections that can 
        be associated with the tracks.

        defaultValue - The weight assigned to tracks and detections that have no connection/edge

        returns - The assignment matrix, a list of row labels, a list of column labels, a 
                  dictionary that locations in the assignment matrix to unique node IDs
        """        

        # Check if the hypothesis is empty
        if (len(self.HYPO) == 0):
            return False, np.ones((0, 0)), {}#np.zeros((0)), np.zeros((0))

        # Create an empty array, filled with the default value, to be populated with the hypothesis
        rowList = sorted(list(self.ALLTRACKS))
        colList = sorted(list(self.ALLDETECTS))
        matrixSize = len(rowList) + len(colList)
        assignmentMatrix = np.ones((matrixSize, matrixSize)) * defaultValue

        # Add false labels to the row and column lists
        for c in sorted(list(self.ALLDETECTS)):
            rowList.append(-c)
        for r in sorted(list(self.ALLTRACKS)):
            colList.append(-r)

        # Set up variables to find the maximum for each row/column
        rowMaxes = [defaultValue] * matrixSize
        colMaxes = [defaultValue] * matrixSize

        # Initialize a lookup dictionary for the array
        nodeLookup = {}
        for i in range(len(rowList)):
            nodeLookup[i] = {}

        # Populate the matrix by adding the weight for every node
        for nodeID in self.HYPO:
            nodeWeight = self.HYPO[nodeID].getWeight()
            nodeTrack = self.HYPO[nodeID].getTreeID()
            nodeDetect = self.HYPO[nodeID].getDetectID()

            # Case where the track is not associated with a detection
            if (nodeDetect == 0):
                track = [nodeTrack]
                detection = [-nodeTrack]
                vals = [nodeWeight]
                nodes = [nodeID]

            # Case where the detection is not associated with a track
            elif (nodeDetect < 0):
                track = [nodeDetect]
                detection = [-nodeDetect]
                vals = [nodeWeight]
                nodes = [nodeID]

            # Case where the track is associated with a detection
            else:
                track = [nodeTrack, -nodeDetect]
                detection = [nodeDetect, -nodeTrack]
                vals = [nodeWeight, 0]
                nodes = [nodeID, None]

            # Populate the matrix
            for t, d, v, n in zip(track, detection, vals, nodes):
                r = rowList.index(t)
                c = colList.index(d)
                if (assignmentMatrix[r, c] < v):
                    assignmentMatrix[r, c] = v
                    nodeLookup[r][c] = n
                    rowMaxes[r] = max(rowMaxes[r], v)
                    colMaxes[c] = max(colMaxes[c], v)

        # Check if the matrix is valid (every row and every column must have at least one entry)
        if ((min(rowMaxes) == defaultValue) or (min(colMaxes) == defaultValue)):
            return False, assignmentMatrix, nodeLookup
        return True, assignmentMatrix, nodeLookup


    def makeHardChoice(self, nodeID):
        """
        Make a hard choice on a node. This means that both the associated track and detection are
        removed from contention, and it is assumed that the node will be chosen regardless of other
        possible associations.

        nodeID - The identifier for the node which a hard decision should be made on

        returns - A new instance of HypoGraph with the updated hypothesis, harddecisions and weight
        """

        # Remove the chosen node from the hypothesis and get its track and detection IDs
        newHypo = self.HYPO.copy()
        removedNode = newHypo.pop(nodeID)
        trackID = removedNode.getTreeID()
        detectID = removedNode.getDetectID()
        weight = removedNode.getWeight()

        # Remove the track and detection ID from the list of all IDs and update the weight for this
        # hypothesis as well as the hard decisions for it
        newWeight = self.CURRENTWEIGHT + weight
        newDecisions = self.HARDDECISIONS + [removedNode]
        newTrackSet = self.ALLTRACKS - {trackID}
        detectID = abs(detectID)
        newDetectSet = self.ALLDETECTS - {detectID}

        # Remove other nodes that share a track or detection
        nodeIDList = list(newHypo.keys())
        for nodeID in nodeIDList:
            if ((trackID == newHypo[nodeID].getTreeID()) or 
                    ((detectID != 0) and (detectID == abs(newHypo[nodeID].getDetectID())))):
                newHypo.pop(nodeID)

        return HypoGraph(newHypo, newDecisions, newTrackSet, newDetectSet, newWeight)


    def removeNode(self, nodeID):
        """
        Remove a node from this hypothesis

        nodeID - The id for the node that should be removed
        """

        # Make a copy of the hypothesis and remove the given node
        if (nodeID in self.HYPO):
            newHypo = self.HYPO.copy()
            
            # Remove the node and return a new instance of the class with the updated hypothesis
            newHypo.pop(nodeID)
            return HypoGraph(newHypo, self.HARDDECISIONS, self.ALLTRACKS, self.ALLDETECTS, 
                             self.CURRENTWEIGHT)

        # Could not find the given track-detection combo
        print("Could not find node:", nodeID, track)
        print(self.HYPO)
        exit()


    def getWeight(self):
        """
        Returns the combined weight for all hard decisions
        """

        return self.CURRENTWEIGHT


    def getHardDecisions(self):
        """
        Returns a list of hard decisions for this hypothesis in the form: 
        [(nodeID1, (track1, detection1)), (nodeID2, (track2, detection2)), ...]
        """

        return self.HARDDECISIONS


    def isEmpty(self):
        """
        Check to see if the hypothesis still has any nodes

        returns - True if the hypothesis is empty (has no nodes) and False if there are any nodes
        """

        return (len(self.HYPO) == 0)


    def hasNode(self, nodeID):
        """
        Check to see if the hypothesis contains the given node

        nodeID - The ID for the node that is being looked up
        """

        return (nodeID in self.HYPO)


    def getSolution(self, nodeList):
        """
        Returns a list of the solutions, given the chosen nodes and including all previous hard 
        decisions

        nodeList - A list of nodeIDs that were selected from the current hypotheses

        returns - The cost of the track-detection combos follwed by a list of decision (including 
                  previous hard decisions) and then the number of associations that correspond to
                  actual detections
        """

        # Create variables to hold the new weight and decisions
        totalWeight = self.CURRENTWEIGHT
        decisions = [] + self.HARDDECISIONS

        # Check to see if every node is valid and update the weight and decisions
        for nodeID in nodeList:
            if (nodeID is None):
                continue
            if (nodeID in self.HYPO):
                chosenNode = self.HYPO[nodeID]
                totalWeight += chosenNode.getWeight()
                decisions.append(chosenNode)

            # The given node ID is not in the hypothesis
            else:
                print("Invalid node selection:", nodeID)
                print()
                print([n.getNodeID() for n in self.HYPO])
                exit()

        return (totalWeight, decisions)


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True, linewidth=np.inf)
    hypo = HypoGraph()
    #nodeID, detectionID, treeID, parent=None, children=[], data=None, weight=0, numMiss=0, endsTree=False
    nodeA = Tree(1, 1, 1, weight=-5.52146)
    nodeB = Tree(2, -1, 1, weight=-10.8198)
    nodeC = Tree(3, 0, 1, weight=0)
    nodeD = Tree(4, 2, 2, weight=-5.52146)
    nodeE = Tree(5, -2, 2, weight=-10.8198)
    nodeF = Tree(6, 0, 2, weight=0)
    nodeG = Tree(7, 1, 3, weight=-22.1914)
    nodeH = Tree(8, 2, 3, weight=-22.1392)
    nodeI = Tree(9, 0, 3, weight=-26.0656)
    nodeJ = Tree(10, 0, 3, weight=-58.3018)
    nodeK = Tree(11, 1, 4, weight=-22.1392)
    nodeL = Tree(12, 2, 4, weight=-22.1914)
    nodeM = Tree(13, 0, 4, weight=-26.0656)
    nodeN = Tree(14, 0, 4, weight=-58.3018)
    hypo.addNode(nodeA)
    hypo.addNode(nodeB)
    hypo.addNode(nodeC)
    hypo.addNode(nodeD)
    hypo.addNode(nodeE)
    hypo.addNode(nodeF)
    hypo.addNode(nodeG)
    hypo.addNode(nodeH)
    hypo.addNode(nodeI)
    hypo.addNode(nodeJ)
    hypo.addNode(nodeK)
    hypo.addNode(nodeL)
    hypo.addNode(nodeM)
    hypo.addNode(nodeN)
    print("#"*100)
    print("First Assignment Matrix")
    valid, mat, lookup = hypo.computeAssignmentMatrix(-1e16)
    print()
    print(valid)
    print(mat)
    print("Lookup:", lookup)
    print()
    weight, decisions, numDetects = hypo.getSolution([1, 4, 9, 13, None, None])
    print("Solution for (1, 4, 9, 13):", [(n.getTreeID(), n.getDetectID()) for n in decisions])
    print("Solution weight:", weight)
    print("#"*100)

    print("#"*100)
    print("Removing node 1 (1 -> 1)")
    hypoa = hypo.removeNode(1)
    valid1, mat1, lookup1 = hypoa.computeAssignmentMatrix(-1e16)
    print()
    print(valid1)
    print(mat1)
    print()
    print(mat - mat1)
    print("Lookup:", lookup1)
    print()
    weight, decisions, numDetects = hypoa.getSolution([3, 4, 7, 13, None, None])
    print("Solution for (3, 4, 7, 13):", [(n.getTreeID(), n.getDetectID()) for n in decisions])
    print("Solution weight:", weight)
    print("#"*100)

    print("#"*100)
    print("Removing node 13 (4 -> 0)")
    hypoa = hypoa.removeNode(13)
    valid2, mat2, lookup2 = hypoa.computeAssignmentMatrix(-1e16)
    print()
    print(valid2)
    print(mat2)
    print()
    print(mat - mat2)
    print("Lookup:", lookup2)
    print()
    print("#"*100)

    print("#"*100)
    print("Removing node 9 (3 -> 0)")
    hypoa = hypoa.removeNode(9)
    valid3, mat3, lookup3 = hypoa.computeAssignmentMatrix(-1e16)
    print()
    print(valid3)
    print(mat3)
    print()
    print(mat - mat3)
    print("Lookup:", lookup3)
    print()
    print("#"*100)

    print("#"*100)
    print("Removing node 3 (1 -> 0)")
    hypoa = hypoa.removeNode(3)
    valid4, mat4, lookup4 = hypoa.computeAssignmentMatrix(-1e16)
    print()
    print(valid4)
    print(mat4)
    print()
    print(mat - mat4)
    print("Lookup:", lookup4)
    print()
    print("#"*100)

    print("#"*100)
    print("Make hard choice on 1 (1 -> 1)")
    hypob = hypo.makeHardChoice(1)
    valid5, mat5, lookup5 = hypob.computeAssignmentMatrix(-1e16)
    print()
    print(valid5)
    print(mat5)
    print("Lookup:", lookup5)
    print("Hard choices:", [n.getNodeID() for n in hypob.getHardDecisions()])
    print("Weight      :", hypob.getWeight())
    print()
    weight, decisions, numDetects = hypob.getSolution([6, 9, 12, None])
    print("Solution for (6, 9, 12):", [(n.getTreeID(), n.getDetectID()) for n in decisions])
    print("Solution weight:", weight)
    print("#"*100)

    print("#"*100)
    print("Make hard choice on 13 (4 -> 0)")
    hypob = hypob.makeHardChoice(13)
    valid6, mat6, lookup6 = hypob.computeAssignmentMatrix(-1e16)
    print()
    print(valid6)
    print(mat6)
    print("Lookup:", lookup6)
    print("Hard choices:", [n.getNodeID() for n in hypob.getHardDecisions()])
    print("Weight      :", hypob.getWeight())
    print()
    weight, decisions, numDetects = hypob.getSolution([6, 9, 5])
    print("Solution for (6, 9, 5):", [(n.getTreeID(), n.getDetectID()) for n in decisions])
    print("Solution weight:", weight)
    print("#"*100)

    print("#"*100)
    print("Make hard choice on 5 (2 -> -2)")
    hypob = hypob.makeHardChoice(5)
    valid6, mat6, lookup6 = hypob.computeAssignmentMatrix(-1e16)
    print()
    print(valid6)
    print(mat6)
    print("Lookup:", lookup6)
    print("Hard choices:", [n.getNodeID() for n in hypob.getHardDecisions()])
    print("Weight      :", hypob.getWeight())
    print()
    weight, decisions, numDetects = hypob.getSolution([9])
    print("Solution for (9):", [(n.getTreeID(), n.getDetectID()) for n in decisions])
    print("Solution weight:", weight)
    print("#"*100)

    print("#"*100)
    print("First Assignment Matrix")
    valid7, mat7, lookup7 = hypo.computeAssignmentMatrix(-1e16)
    print()
    print(valid7)
    print(mat7)
    print()
    print(mat - mat7)
    print("Lookup:", lookup7)
    print()
    print("#"*100)
