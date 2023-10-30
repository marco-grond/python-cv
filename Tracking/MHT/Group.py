#!/usr/bin/env python

###################################################################################################
#                                           Group Class                                           #
#                                                                                                 #
# Class for creating and maintaining groups along with group hypotheses that are used as an       #
# implementation of the multiple hypothesis tracker. This class allows for creating a new group   #
# with leaves of a tree, merging and splitting groups so that the trees in the group are all      #
# related in some way and generating and maintaining group hypotheses to track objects by         #
# delaying data association.                                                                      #
#                                                                                                 #
# Author: Marco Grond                                                                             #
# Version: 0.1.0                                                                                  #
###################################################################################################

import heapq
from Tree import Tree
from TrackTree import TrackTree
from Assign import assignment
import Model
from copy import deepcopy
import numpy as np
from HypoGraph import HypoGraph

class Group:

    def __init__(self, maxHypotheses, minHypothesisRatio, initialGroup, groupHypos, nScan, maxTime, 
                 debug=False):
        """
        Initialize an instance of the group class, which is used when computing the global 
        hypothesis for the multiple hypothesis tracker

        maxHypotheses - The maximum number of group hypotheses that are allowed
        minHypothesisRatio - The minimum ratio allowed between the largest and smallest likelihoods
                             for group hypotheses
        initialGroup - A dictionary with the group information. The keys are the tree numbers, 
                       while the values for each key is a list of the leaf nodes for the tree, e.g.
                       {treeID1: [leafNode1, leafNode2, ...], ...}
        groupHypos - A list of predetermined group hypotheses and their respective likelihoods, in 
                     the form: [[likelihood1, [leaf1, leaf2, ...]], ...]
        nScan - The allowed depth of a track tree before pruning has to take place
        maxTime - The maximum time difference that is allowed before a tree is ended
        debug - A flag for printing debugging statements
        """

        self.HYPOTHESES = []
        self.MAXHYP = maxHypotheses
        self.MINHYPRATIO = minHypothesisRatio
        self.GROUP = initialGroup
        self.ASSIGN = assignment([], [], [], maximize=True)
        self.DEFAULTVALUE = self.ASSIGN.getDefaultValue()
        self.NSCAN = nScan
        self.MAXT = maxTime
        self.DEBUG = debug
        for h in groupHypos:
            heapq.heappush(self.HYPOTHESES, h)


    def checkAndSplit(self, treeGroups):
        """
        Checks if this group should be split into multiple groups. Returns a list containing the 
        new groups if they were split, otherwise just returns a list with a copy of itself

        treeGroups - A dictionary where the keys are the tree IDs and the values are the group
                     number associated with a tree

        returns - A list of all newly split groups. If the group should not be split, a list with
                  just this group is returned
        """

        # Check if the group is empty
        if (len(self.GROUP) == 0):
            return []

        # Go through all of the trees in the group, and check if they have different group numbers
        groupTrees = list(self.GROUP.keys())
        groupNums = [treeGroups[groupTrees[0]]]
        allSame = True
        possibleGroups = [set()]
        for tree in groupTrees:
            nextGroup = treeGroups[tree]
            if (nextGroup in groupNums):
                index = groupNums.index(nextGroup)
                possibleGroups[index].add(tree)
            else:
                allSame = False
                groupNums.append(nextGroup)
                possibleGroups.append({tree})

        # Split this group, as well as its group hypotheses into new groups if necessary
        if allSame:
            return [self]
        returnList = []
        for newGroup in possibleGroups:
            groupDict = {}
            for tree in newGroup:
                groupDict[tree] = self.GROUP.pop(tree)
            newHypotheses = []
            allUnique = []
            groupTreeIDs = list(groupDict.keys())
            for _, hypo in self.getGroupHypotheses():
                groupHypos = []
                likelihood = 0
                hypoRepr = np.ones(len(groupTreeIDs)).astype(int)*-1
                for leaf in hypo:
                    if (leaf.getTreeID() in newGroup):
                        groupHypos.append(leaf)
                        likelihood += leaf.getWeight()
                        hypoRepr[groupTreeIDs.index(leaf.getTreeID())] = leaf.getNodeID()

                # Check to see if the group hypothesis is repeated, and ignore if it is
                shouldAdd = True
                for unique in allUnique:
                    if (np.all(hypoRepr == unique)):
                        shouldAdd = False
                if shouldAdd:
                    newHypotheses.append([likelihood, groupHypos])
                    allUnique.append(hypoRepr)

            # Create a new group with the given trees and hypotheses, and save it to be returned
            returnList.append(Group(self.MAXHYP, self.MINHYPRATIO, groupDict, newHypotheses, 
                                    self.NSCAN, self.MAXT, self.DEBUG))

        return returnList


    def merge(self, newGroup):
        """
        Merge this group, along with its group hypotheses, with the provided group. This operation
        is done in place, so only this group should be kept (newGroup can be discarded after this
        operation is performed)

        group - The group that this group should be merged with
        """

        # Initialize sorted lists of group hypotheses and a new structure to store the combined 
        # hypotheses
        thisHyp = self.getGroupHypotheses()
        otherHyp = newGroup.getGroupHypotheses()
        newHypotheses = []
        indexPairs = []
        largestLikelihood = thisHyp[0][0] + otherHyp[0][0]
        heapq.heappush(indexPairs, (largestLikelihood, 0, 0))
        flagList = [0]*(len(thisHyp)*len(otherHyp))
        flagList[0] = 1

        if self.DEBUG:
            print("----------------------------- Merging Groups ----------------------------")
            print("First Group Hypotheses:")
            self.printGroupHypotheses()
            print("Second Group Hypotheses:")
            newGroup.printGroupHypotheses()
            print('-'*74)

        # Find the combined hypotheses, in decreasing order of likelihood, to be added onto the 
        # new group hypotheses priority queue while conforming to the rules set out for the group 
        # hypotheses
        while ((len(indexPairs) > 0) and (len(newHypotheses) <= self.MAXHYP)):

            # Check to see if the likelihood of the combined hypotheses isn't too small and add it
            # to the new group hypotheses list if not
            likelihood, thisIndex, otherIndex = heapq.heappop(indexPairs)
            if ((largestLikelihood - likelihood) <= self.MINHYPRATIO):
                heapq.heappush(newHypotheses, 
                               (likelihood, thisHyp[thisIndex][1] + otherHyp[otherIndex][1]))

            # Add the next two possible hypotheses onto the priority queue, if they have not been 
            # added to it before
            if (((thisIndex+1) < len(thisHyp)) and 
                    (flagList[(thisIndex+1)+len(thisHyp)*otherIndex] == 0)):
                flagList[(thisIndex+1)+len(thisHyp)*otherIndex] = 1
                heapq.heappush(indexPairs, (thisHyp[thisIndex+1][0]+otherHyp[otherIndex][0], 
                                            thisIndex+1, otherIndex))

            if (((otherIndex+1) < len(otherHyp)) and 
                    (flagList[thisIndex+len(thisHyp)*(otherIndex+1)] == 0)):
                flagList[thisIndex+len(thisHyp)*(otherIndex+1)] = 1
                heapq.heappush(indexPairs, (thisHyp[thisIndex][0]+otherHyp[otherIndex+1][0], 
                                            thisIndex, otherIndex+1))

        # Update the priority queue as well as the dictionary of trees
        self.HYPOTHESES = newHypotheses
        self.GROUP.update(newGroup.getGroupTrees())

        if self.DEBUG:
            print("--------------------------- Merged Hypotheses ---------------------------")
            self.printGroupHypotheses()
            print('-'*74)


    def getGroupTrees(self):
        """
        Returns a dictionary containing all trees in the group
        """

        return self.GROUP


    def getGroupTreeIDs(self):
        """
        Returns a list containing all of the IDs for trees in this group
        """

        return list(self.GROUP.keys())


    def getGroupHypotheses(self):
        """
        Returns the list of group hypotheses for this group, sorted from largest to smallest weight
        """

        return heapq.nlargest(self.MAXHYP, self.HYPOTHESES)


    def isEmpty(self):
        """
        Checks to see if the current group is empty (there are no trees or no group hypotheses)

        returns - True if the group is empty and False if the group is not empty
        """

        return ((len(self.GROUP) == 0) or (len(self.HYPOTHESES) == 0))


    def hasTree(self, treeNum):
        """
        Check to see if the given treeNum is in the group

        treeNum - The unique identifier for the tree
        """

        return (treeNum in self.GROUP)


    def removeTree(self, treeNum):
        """
        Remove the given tree number from the group and updates all group hypotheses by removing 
        references to this tree

        treeNum - The number of the tree that should be removed

        returns - True if there are still trees/grouphypotheses remaining in this group, False if 
                  the group is empty and should be discarded
        """

        if (not treeNum in self.GROUP):
            return self.isEmpty()

        # Remove the tree from the group and update the group hypotheses
        self.GROUP.pop(treeNum)
        if (len(self.GROUP) == 0):
            return False
        newHypotheses = []
        allUnique = []
        groupTreeIDs = list(self.GROUP.keys())
        for _, hypo in self.getGroupHypotheses():
            groupHypos = []
            likelihood = 0
            hypoRepr = np.ones(len(groupTreeIDs)).astype(int)*-1
            for leaf in hypo:
                if (leaf.getTreeID() != treeNum):
                    groupHypos.append(leaf)
                    likelihood += leaf.getWeight()
                    hypoRepr[groupTreeIDs.index(leaf.getTreeID())] = leaf.getNodeID()

            # Check to see if the group hypothesis is repeated, and ignore if it is
            shouldAdd = True
            for unique in allUnique:
                if (np.all(hypoRepr == unique)):
                    shouldAdd = False
            if shouldAdd:
                heapq.heappush(newHypotheses, [likelihood, groupHypos])
                allUnique.append(hypoRepr)

        if (len(newHypotheses) == 0):
            return False
        self.HYPOTHESES = newHypotheses
        return True


    def createAssignmentGraph(self, groupHypothesis):
        """
        Creates an assignment graph from the given group hypothesis

        groupHypothesis - One group hypothesis consisting of a number of independent tree nodes
        """

        graph = HypoGraph()

        # Go through all leaves in the group hypothesis and add an edge
        for node in groupHypothesis:
            for leaf in node.getLeaves():
                graph.addNode(leaf)

        return graph


    def createAssignmentMatrix(self, groupHypothesis):
        """
        Create an assignment matrix problem for the given group hypothesis

        groupHypothesis - One group hypothesis consisting of a number of independent tree nodes

        returns - (AssignmentMatrix, RowLabelList, ColumnLabelList)
        """

        leafList = []
        detectionSet = set()
        treeSet = set()
        zeroList = []
        onesList = []

        # Go through all of the leaves of each node in the group hypothesis
        for node in groupHypothesis:
            treeSet.add(node.getTreeID())
            newLeaves = node.getLeaves()
            leafList += newLeaves

            # Find all unique Node IDs associated with this tree
            for leaf in newLeaves:
                # Case where the node is a dummy or end node
                if (leaf.getDetectID() == 0):
                    zeroList.append(leaf.getTreeID())

                # Case where the node is a false alarm node
                elif (leaf.getDetectID() == 1):
                    onesList.append(leaf.getTreeID())

                # Case where the node is a detection or skip node (all are unique)
                else:
                    detectionSet.add(leaf.getDetectID())
                treeSet.add(leaf.getTreeID())

        # Set up matrices to hold values for all of the different kinds of nodes
        trees = list(treeSet)
        detections = list(detectionSet)
        endMatrix = np.ones((len(trees), len(zeroList))) * self.DEFAULTVALUE
        fAlarmMatrix = np.ones((len(trees), len(onesList))) * self.DEFAULTVALUE
        assignmentMatrix = np.ones((len(trees), len(detectionSet))) * self.DEFAULTVALUE

        # For each leaf, enter its weight into the correct matrix
        for leaf in leafList:
            row = trees.index(leaf.getTreeID())
            if (leaf.getDetectID() == 0):
                col = zeroList.index(leaf.getTreeID())
                endMatrix[row][col] = leaf.getWeight()
            elif (leaf.getDetectID() == 1):
                col = onesList.index(leaf.getTreeID())
                fAlarmMatrix[row][col] = leaf.getWeight()
            else:
                col = detections.index(leaf.getDetectID())
                assignmentMatrix[row][col] = leaf.getWeight()#leaf.getData().getScore()

        # Add columns to assignment matrix for the case that a tree should be discarded
        onesCount = len(onesList)
        if (onesCount > 0):
            assignmentMatrix = np.concatenate((assignmentMatrix, fAlarmMatrix), axis=1)
            detections += [1]*onesCount
        zeroCount = len(zeroList)
        if (zeroCount > 0):
            assignmentMatrix = np.concatenate((assignmentMatrix, endMatrix), axis=1)
            detections += [0]*zeroCount

        return assignmentMatrix, trees, detections


    def cleanAssignment(self, assignment):
        """
        Cleans the assignment to make sure that there aren't any clashes (can occur when
        dummy detections are present) and to remove false row/col associations

        assignment - A list of assignments generated for a group hypothesis

        returns - A list containing the updated assignements
        """

        # Associate all nodes with the correct tree ID
        trees = {}
        for node in assignment:
            treeID = node.getTreeID()
            if (not treeID in trees):
                trees[treeID] = []
            trees[treeID].append(node)

        # Remove clashes from the dictionary
        treeIDs = list(trees.keys())
        returnList = []
        for tree in treeIDs:

            # Choose the option with the lowest weight
            if (len(trees[tree]) > 1):
                options = trees[tree]
                chosenOption = options[0]
                chosenWeight = chosenOption.getWeight()
                for opt in options[1:]:
                    if (opt.getWeight() < chosenWeight):
                        chosenOption = opt
                        chosenWeight = opt.getWeight()
                returnList.append(chosenOption)
            else:
                returnList.append(trees[tree][0])

        return returnList


    def generateGroupHypotheses(self, trackTrees, time):
        """
        Generate the set of group hypotheses, until the maximum number of allowed group hypotheses 
        have been reached, for the current group.

        trackTrees - A dictionary of all of the current track trees, where the key is the tree
                     number and the value is the corresponding TrackTree object
        time - The time at which this operation is performed

        Returns - A set of all TrackTrees that should be ended, and a list containing all nodes 
                  that were pruned
        """

        # Set up variables to store the assignment matrices for each of the group hypotheses
        hypoList = heapq.nlargest(len(self.HYPOTHESES), self.HYPOTHESES)
        graphList = []
        if self.DEBUG:
            print("\n-------------------- Assignment Matrices and Labels ---------------------")
            print("Trees in group:", self.getGroupTreeIDs())
            print("Previous hypotheses:")
            self.printGroupHypotheses()
            print()

        # Go through all group hypotheses and create assignment graphs for each            
        for _, hypo in hypoList:
            graphList.append(self.createAssignmentGraph(hypo))
            if self.DEBUG:
                valid, assignMat, lookup = graphList[-1].computeAssignmentMatrix(-1e16)
                np.set_printoptions(precision=6, suppress=True, linewidth=np.inf)
                print("Assignment Matrix:\n" + str(assignMat))
                print("Is valid:", valid)
                print("Lookup  :", lookup)
                print()

        # Check to see if possible solutions still exist
        assigner = self.ASSIGN.setQueue(graphList)
        if (not self.ASSIGN.hasSolution()):
            return

        # Find the best solution, make a new group hypothesis for it and create the remaining group
        # hypotheses
        bestCost, bestAssignment, bestIndex = self.ASSIGN.getBest(-np.inf)
        bestAssignment = self.cleanAssignment(bestAssignment)
        bestHypo = hypoList[bestIndex][1]
        cost = bestCost
        newGroupHypos = []
        if self.DEBUG:
            print("-------------------------- Cost and Assignment --------------------------")
            print("Best Cost:", bestCost)
            print("Best Assignment:", bestAssignment)
            print("Best Index:", bestIndex)
            print()
            counter = 1
        while ((self.ASSIGN.hasSolution()) and (len(newGroupHypos) < self.MAXHYP)):
            cost, assignment, index = self.ASSIGN.getBest((bestCost - self.MINHYPRATIO))
            if ((bestCost - cost) > self.MINHYPRATIO):
                break
            assignment = self.cleanAssignment(assignment)
            if self.DEBUG:
                print("Cost " + str(counter) + ": " + str(cost))
                print("Assignment " + str(counter) + ": " + str(assignment))
                print("Index:", index)
                print()
                counter += 1
            newGroupHypos.append([cost, assignment, index])

        # Prune the trees, w.r.t. the best hypothesis, and remove all group hypotheses that lose 
        # leaves and end trees that aren't updated
        if self.DEBUG:
            print("\nTrees before pruning:")
            for tree in self.getGroupTreeIDs():
                trackTrees[tree].getRoot().printTree()
            print()

        removeSet = set()
        allTrees = set(self.getGroupTreeIDs())
        removedLeaves = {tree:[] for tree in allTrees}
        updatedBest = []
        for bestNode in bestAssignment:
            tree = bestNode.getTreeID()
            nodeID = bestNode.getNodeID()
            allTrees.remove(tree)

            if (trackTrees[tree].getDepth() > self.NSCAN):

                # This tree was chosen as part of the solution, so prune leaves that diverge from
                # the hypothesis
                for node in bestHypo:
                    if (node.getTreeID() == tree):

                        # Check to see if this tree was not chosen as part of the solution (should be 
                        # discarded)
                        if (bestNode.shouldEnd()):
                            removeSet.add(tree)
                            break
                        updatedBest.append(bestNode)
                        chosenLeaf = bestNode
                        for i in range(self.NSCAN-1):
                            chosenLeaf = chosenLeaf.getParent()
                        _, prunedLeaves = trackTrees[tree].keep(
                                                           chosenLeaf.getNodeID(), time)
                        removedLeaves[tree] = prunedLeaves
                        break
            else:
                updatedBest.append(bestNode)

        # Delete trees that were not in the best hypothesis from this group's dictionary of trees
        for tree in allTrees:
            if ((time - trackTrees[tree].getLastUpdate()) > self.MAXT):
                removeSet.add(tree)
        for tree in removeSet:
            if tree in self.GROUP:
                self.GROUP.pop(tree)

        # Save hypotheses that do no not have a removed detection or tree to the priority queue
        self.HYPOTHESES = [[bestCost, updatedBest]]
        for cost, assignment, index in newGroupHypos:
            keep = True
            updatedAssignment = []
            for node in assignment:
                tree = node.getTreeID()
                nodeID = node.getNodeID()
                if ((tree in removeSet) or (nodeID in removedLeaves[tree])):
                    keep = False
                    break
                updatedAssignment.append(node)
            if (keep and (len(updatedAssignment) > 0)):
                heapq.heappush(self.HYPOTHESES, [cost, updatedAssignment])

        # Go through the current hypothesis list and remove all leaves that aren't in any hypotheses
        allLeaves = []
        onlyEndTrees = set(self.GROUP.keys())
        for treeID in self.GROUP:
            allLeaves += trackTrees[treeID].getLeaves()
        allNodeIDs = set([l.getNodeID() for l in allLeaves])
        for cost, leaves in self.HYPOTHESES:
            for leaf in leaves:
                allNodeIDs.discard(leaf.getNodeID())
        for leaf in allLeaves:
            if (leaf.getNodeID() in allNodeIDs):
                leaf.removeThisLeaf()
            elif (not leaf.shouldEnd()):
                onlyEndTrees.discard(leaf.getTreeID())
        for tree in onlyEndTrees:
            self.removeTree(tree)
            removeSet.add(tree)

        if self.DEBUG:
            print("Trees after pruning:")
            for tree in self.getGroupTreeIDs():
                trackTrees[tree].getRoot().printTree()
            print()
            print("Trees that should be discarded:", removeSet)
            print("Remaining Hypotheses:")
            self.printGroupHypotheses() 
            print('-'*70)

        return removeSet


    def createGroupHypos(self, trackTrees, assignment, previousHypothesis):
        """
        Creates and returns a group hypothesis from the given assignment and trackTrees, so that
        each leaf in the group hypothesis satisfies one of the assignments.

        trackTrees - A dictionary of all of the current track trees, where the key is the tree
                     number and the value is the corresponding TrackTree object
        assignment - A list of tuples in the form (tree, detection), where tree is the ID of the 
                     tree that the detection ID is assigned to
        previousHypothesis - The hypothesis that the new hypothesis is based on

        Returns - A list of leaf nodes which acts as a group hypothesis
        """

        # Go through each assignment and add the corresponding leaf from the trackTree to the hypo
        newHypo = []
        for nodeID, (tree, detection) in assignment:
            for previousLeaf in previousHypothesis:
                if (previousLeaf.getTreeID() == tree):
                    newHypo.append(previousLeaf.getChild(nodeID))

        return newHypo


    def printGroupHypotheses(self):
        """
        Prints the individual group hypotheses for this group (tree -> detection)
        """

        sortedHypos = self.getGroupHypotheses()
        for cost, gh in sortedHypos:
            print("Cost:", cost, "-", end="")
            for n in gh:
                print("(" + str(n.getDetectID()) + ", (" + str(n.getTreeID()) + " -> " + str(n.getNodeID()) + ")) ", end="")
            print()



if __name__ == '__main__':

    # Set up basic detections
    baseCov = np.array([[3, 0], [0, 3]])
    detectA = np.array([[0], [0]])
    detectB = np.array([[1], [1]])
    detectC = np.array([[1], [0]])
    detectD = np.array([[2], [0]])
    detectE = np.array([[2], [2]])
    detectF = np.array([[3], [0]])
    detectG = np.array([[3], [3]])
    detectH = np.array([[4], [0]])
    detectI = np.array([[4], [4]])
    detectJ = np.array([[2.5], [0]])
    detectK = np.array([[3.5], [3.5]])

    detect1 = np.array([[10], [10]])
    detect2 = np.array([[10], [12]])
    detect3 = np.array([[10], [8]])
    detect4 = np.array([[10], [14]])
    detect5 = np.array([[10], [9]])
    detect6 = np.array([[10], [16]])
    detect7 = np.array([[10], [1]])
    detect8 = np.array([[10], [18]])
    detect9 = np.array([[10], [10]])

    # Set up basic models
    A = Model.Kalman([detectA, baseCov], 0)
    B = deepcopy(A)
    B.update([detectB, baseCov], 1)
    E = deepcopy(B)
    E.update([detectE, baseCov], 2)
    G = deepcopy(E)
    G.update([detectG, baseCov], 3)
    I = deepcopy(G)
    I.update([detectI, baseCov], 4)
    C = deepcopy(A)
    C.update([detectC, baseCov], 1)
    D = deepcopy(C)
    D.update([detectD, baseCov], 2)
    F = deepcopy(D)
    F.update([detectF, baseCov], 3)
    H = deepcopy(F)
    H.update([detectH, baseCov], 4)
    J = deepcopy(D)
    J.update([detectJ, baseCov], 3)
    K = deepcopy(E)
    K.update([detectK, baseCov], 3)

    M1 = Model.Kalman([detect1, baseCov], 0)
    M2 = deepcopy(M1)
    M2.update([detect2, baseCov], 1)
    M4 = deepcopy(M2)
    M4.update([detect4, baseCov], 2)
    M6 = deepcopy(M4)
    M6.update([detect6, baseCov], 3)
    M8 = deepcopy(M6)
    M8.update([detect8, baseCov], 4)
    M3 = deepcopy(M1)
    M3.update([detect3, baseCov], 1)
    M5 = deepcopy(M3)
    M5.update([detect5, baseCov], 2)
    M7 = deepcopy(M5)
    M7.update([detect7, baseCov], 3)
    M9 = deepcopy(M7)
    M9.update([detect9, baseCov], 4)


    # Create tree nodes
    nodeA = Tree(nodeID=0, treeID=0, parent=None,  children=[], data=A)
    node1 = Tree(nodeID=1, treeID=1, parent=None,  children=[], data=M1)

    # Create group hypotheses
    #hypo0 = [[nodeB.getData().getScore(), [nodeB]], [nodeC.getData().getScore(), [nodeC]], [nodeD.getData().getScore(), [nodeD]]]
    #hypo1 = [[nodeF.getData().getScore(), [nodeF]], [nodeG.getData().getScore(), [nodeG]], [nodeH.getData().getScore(), [nodeH]]]
    #hypo2 = [[nodeJ.getData().getScore(), [nodeJ]], [nodeK.getData().getScore(), [nodeK]], [nodeL.getData().getScore(), [nodeL]]]
    hypo0 = [[nodeA.getData().getScore(), [nodeA]]]
    hypo1 = [[node1.getData().getScore(), [node1]]]

    # Create groups
    Group0 = Group(maxHypotheses=30,
                   minHypothesisRatio=1,
                   initialGroup={0:nodeA},
                   groupHypos=hypo0,
                   nScan=3,
                   maxTime=5,
                   debug=True)
    Group1 = Group(maxHypotheses=30,
                   minHypothesisRatio=1,
                   initialGroup={1:node1},
                   groupHypos=hypo1,
                   nScan=3,
                   maxTime=5,
                   debug=True)
    print("Group 0")
    Group0.printGroupHypotheses()
    print("\nGroup 1")
    Group1.printGroupHypotheses()

    # Merge groups
    print("\nMerging groups 0 and 1")
    Group0.merge(Group1)
    Group0.printGroupHypotheses()

    # Split groups
    treeGroups = {0:0, 1:1, 2:0}
    print("\nSplitting groups")
    newGroups = Group0.checkAndSplit(treeGroups)
    print("Number of new Groups:", len(newGroups))
    for i, group in enumerate(newGroups):
        print("Group", i)
        group.printGroupHypotheses()
        print()

    # Test hypothesis generation
    print("*"*50)
    print("\nHypothesis generation")
    print("*"*50)
    tt = {0:TrackTree(nodeA, 3, 0), 1:TrackTree(node1, 3, 0)}

    nodeB = Tree(nodeID=0, treeID=0, parent=nodeA, children=[], data=B)
    nodeC = Tree(nodeID=1, treeID=0, parent=nodeA, children=[], data=C)
    node2 = Tree(nodeID=2, treeID=1, parent=node1, children=[], data=M2)
    node3 = Tree(nodeID=3, treeID=1, parent=node1, children=[], data=M3)
    print("#"*80)
    print("#"*80)
    print("Time 1:")
    for i, group in enumerate(newGroups):
        print("Group", i)
        group.generateGroupHypotheses(tt, 1)
        group.printGroupHypotheses()
        print()
        print("#"*80)
    print("#"*80)

    nodeD = Tree(nodeID=0, treeID=0, parent=nodeC, children=[], data=D)
    nodeE = Tree(nodeID=1, treeID=0, parent=nodeB, children=[], data=E)
    node4 = Tree(nodeID=2, treeID=1, parent=node2, children=[], data=M4)
    node5 = Tree(nodeID=3, treeID=1, parent=node3, children=[], data=M5)
    print("Time 2:")
    for i, group in enumerate(newGroups):
        print("Group", i)
        group.generateGroupHypotheses(tt, 1)
        group.printGroupHypotheses()
        print()
    print("*"*50)

    nodeF = Tree(nodeID=0, treeID=0, parent=nodeD, children=[], data=F)
    nodeJ = Tree(nodeID=4, treeID=0, parent=nodeD, children=[], data=J)
    nodeG = Tree(nodeID=1, treeID=0, parent=nodeE, children=[], data=G)
    nodeK = Tree(nodeID=5, treeID=0, parent=nodeE, children=[], data=K)
    node6 = Tree(nodeID=2, treeID=1, parent=node4, children=[], data=M6)
    node7 = Tree(nodeID=3, treeID=1, parent=node5, children=[], data=M7)
    print("Time 3:")
    for i, group in enumerate(newGroups):
        print("Group", i)
        group.generateGroupHypotheses(tt, 1)
        group.printGroupHypotheses()
        print()
    print("*"*50)

    nodeH = Tree(nodeID=0, treeID=0, parent=nodeF, children=[], data=H)
    nodeI = Tree(nodeID=1, treeID=0, parent=nodeG, children=[], data=I)
    node8 = Tree(nodeID=2, treeID=1, parent=node6, children=[], data=M8)
    node9 = Tree(nodeID=3, treeID=1, parent=node7, children=[], data=M9)
    print("Time 4:")
    for i, group in enumerate(newGroups):
        print("Group", i)
        group.generateGroupHypotheses(tt, 1)
        group.printGroupHypotheses()
        print()
    print("*"*50)
    


    #A = Group(maxHypotheses=10,
    #          minHypothesisRatio=0,
    #          initialNode=1,
    #          hypotheses=[(0.1, ['A']), (0.3, ['B']), (0.8, ['C']), (0.2, ['D']), (0.25, ['E']), (0.7, ['F'])])
    #print(A.getGroupHypotheses())
    #B = Group(maxHypotheses=10,
    #          minHypothesisRatio=0,
    #          initialNode=1,
    #          hypotheses=[(0.9, ['G']), (0.5, ['H']), (0.005, ['I']), (0.004, ['J']), (0.003, ['K']), (0.002, ['L'])])
    #print(B.getGroupHypotheses())
    #A.merge(B)
    #print(A.getGroupHypotheses())
