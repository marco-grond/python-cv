#!/usr/bin/env python

###################################################################################################
#                                Multiple Hypothesis Tracker Class                                #
#                                                                                                 #
# Multiple Hypothesis Tracker with added functionality to suspend and restart tracks for objects  #
# that become occluded. Multiple possible data associations are explored for a number of frames   #
# before any decisions are made. Two implementations of a multiple hypothesis tracker are         #
# available.                                                                                      #
#                                                                                                 #
# The first uses hypotheses, groups and assignment matrices and is based on the work              #
# of Ingemar J. Cox and Sunita L. Hingorani in "An Efficient Implementation of Reid's Multiple    #
# Hypothesis Tracking Algorithm and Its Evaluation for the Purpose of Visual Tracking", 1996.     #
#                                                                                                 #
# The second implementation uses the Maximum Weighted Independent Set implementation outlined in  #
# the paper "Multiple hypothesis tracking revisited" by C. Kim et. al. in 2015 and is based on    #
# code from openmht by Jon Perdomo available at https://github.com/jonperdomo/openmht             #
#                                                                                                 #
# Author: Marco Grond                                                                             #
# Version: 0.1.0                                                                                  #
###################################################################################################

import numpy as np
import math
from WeightedGraph import WeightedGraph
from Tree import Tree
from TrackTree import TrackTree
from Group import Group
from copy import deepcopy
from scipy.optimize import linear_sum_assignment as hungarian

import time

class MHT:

    def __init__(self, model, gateDistance, timeDrop, minTrackUpdates, pNewTrack, pFalse, pDetect, 
                 endLambda, nScan, maxHypotheses, minHypothesisRatio, timeInterval=1, modelArgs=(),
                 debug=False):
        """
        Initialize an instance of the Multiple Hypothesis Tracker.

        model - The model to be used when predicting the movement of an object (Kalman)
        gateDistance - The max distance at which a detection may be associated with a track, i.e.
                       the gating distance for a predicted location and object detections
        timeDrop - The time that a track will be kept without an update
        minTrackLen - The minimum number of updates that a track needs to have to be considered 
                      a true track

        ###pTrue - The probability of detecting a track

        pNewTrack - The probability of starting a new track
        pFalse - The probability that a data point will not be assigned
        pDetect - The probability of detecting an object that was previously detected
        endLambda - A constant used to determine the likelihood of a track ending
        nScan - The allowed depth of track trees before they are pruned (the length of time a track
                tree is allowed to diverge before being pruned)
        maxHypotheses - The maximum number of hypotheses that are allowed
        minHypothesisRatio - The minimum ratio allowed between the largest and smallest hypotheses 
                             weights in a single group
        timeInterval - The interval between different detections (for a frame by frame, would be 1
                        and for seconds at 30 fps would be 1/30)
        modelArgs - Additional arguments passed to the model during initialization. When an 
                    instance of the model is initialized, the initial measurement data, time and 
                    these arguments are passed. Arguments should be a tuple.
        """

        # Initialize variables used during the operation of the MHT
        self.MODEL = model(*modelArgs)
        self.GATE = gateDistance
        self.MAXT = timeDrop
        self.TIMEINT = timeInterval

        # Set up variables used for computing weights for nodes and leaves
        self.FALSEALARM = np.log(pFalse)
        self.NEWLOG = np.log(pNewTrack)
        self.DETECTLOG = np.log(pDetect)
        self.SKIPLOG = np.log(1.0 - pDetect)
        self.ENDLAMBDA = endLambda

        # Initialize variables to maintain track trees
        self.NSCAN = nScan

        # Initialize datastructures to store track information
        self.TRACKTREES = {}
        self.TRACKID = 1
        self.NODEID = 1
        self.MINTRACKUPDATES = minTrackUpdates
        self.ENDEDTRACKS = {}

        # Initialize datastructures for suspending, restarting and ending tracks
        self.SUSPENDLIST = {}
        self.POSSIBLERESTART = {}
        self.POSSIBLEEND = {}
        self.RESTARTMAP = {}

        # Initialize datastructures and variables for grouping
        self.GROUPS = []
        self.MAXHYPOS = maxHypotheses
        self.MINHYPRATIO = -np.log(minHypothesisRatio)

        self.DEBUG = debug


    def iterateMWIS(self, detectLocs, t):
        """
        Move all the tracks forward to the given time and associate the given detections with
        existing tracks or create new tracks using a multiple hypothesis tracker

        detectLocs - A list of locations and covariances (x, P) corresponding to object detections
        t - The current time step (i.e. the time step at which the detections took place)
        """

        # Compute groups for all of the track trees
        groups = self.getGroups(t-1)

        # Print Iteration information
        print("*"*100)
        print("Time           :", t)
        print("Num Track Trees:", len(self.TRACKTREES))
        print("Num Groups     :", len(groups))

        if self.DEBUG:
            print("Performing iteration with:")
            print("\tObject detections:", detectLocs)

        # Prune all track trees according to the global hypothesis
        startTime = time.time()
        keep, endedTrees = self.computeGlobalHypothesis(groups)
        endTime = time.time()
        print("Global Hypothesis Time:", (endTime - startTime))
        for tt, keepBranch in keep:
            if not self.TRACKTREES[tt].keep(keepBranch, t):
                print("***WARNING: Attempted to keep non existant branch", keepBranch, 
                      "in tree", tt, "***")
                print("Removing tree...")
                self.TRACKTREES[tt].end()
                ended = self.TRACKTREES.pop(tt)
                if (ended.getNumUpdates() >= self.MINTRACKUPDATES):
                    self.ENDEDTRACKS[tt] = ended

        # End all track trees that need to be removed accoring to the global hypothesis
        for tt in endedTrees:
            self.TRACKTREES[tt].end()
            ended = self.TRACKTREES.pop(tt)
            if (ended.getNumUpdates() >= self.MINTRACKUPDATES):
                self.ENDEDTRACKS[tt] = ended

        # End all track trees that have not been updated recently enough
        treeKeys = list(self.TRACKTREES.keys())
        for tt in treeKeys:
            if (((t - self.TRACKTREES[tt].getLastUpdate() - self.NSCAN*self.TIMEINT) >= self.MAXT)
                    and (self.TRACKTREES[tt].getDepth() >= self.NSCAN)):
                self.TRACKTREES[tt].end()
                ended = self.TRACKTREES.pop(tt)
                if (ended.getNumUpdates() >= self.MINTRACKUPDATES):
                    self.ENDEDTRACKS[tt] = ended

        # Get all possible object locations (all leaves in all track trees)
        leafNodes = []
        for tt in self.TRACKTREES:
            leafNodes += [(tt, leaf) for leaf in self.TRACKTREES[tt].getLeaves()]

        # For each leaf, find all possible detections that could match and create new branches
        for treeID, leaf in leafNodes:
            objectState, lastTime, lastUpdate = leaf.getData()
            objectState = self.MODEL.extrapolate(objectState, (t - lastTime))
            numMissedUpdates = leaf.getNumMissed()
            endLikelihood = max((1.0 - np.exp(-numMissedUpdates/self.ENDLAMBDA)), 1e-14)
            endLog = np.log(endLikelihood)
            contLog = np.log(1 - endLikelihood)

            # For each object detection, see if it is close enough to the leaf node and create a 
            # new branch/leaf in the case that it is
            if (not leaf.shouldEnd()):
                for i, detection in enumerate(detectLocs):
                    if (self.MODEL.distance(detection, objectState) < self.GATE):
                        newState = self.MODEL.update(detection, objectState)
                        nodeWeight = -1/self.MODEL.getMotionScore(detection, newState)
                        Tree(self.NODEID, (i+1), treeID, parent=leaf, children=[], data=(newState, t, t),
                             weight=nodeWeight, numMiss=0, endsTree=False)
                        self.NODEID += 1

            # Add instance where the object might be lost for a frames
            if ((t - lastUpdate) < self.MAXT):
                nodeWeight = -1/self.MODEL.getMotionScore(objectState, objectState)
                Tree(self.NODEID, 0, treeID, parent=leaf, children=[], 
                     data=(objectState, t, lastUpdate), weight=nodeWeight, 
                     numMiss=numMissedUpdates+1, endsTree=False)
                self.NODEID += 1

            # Add instance where the track should end
            else:
                nodeWeight = leaf.getWeight()
                Tree(self.NODEID, 0, treeID, parent=leaf, children=[], 
                     data=(objectState, t, lastUpdate), weight=nodeWeight, 
                     numMiss=numMissedUpdates+1, endsTree=True)
                self.NODEID += 1    

        # Create a new root node for every object detection
        for i, detection in enumerate(detectLocs):
            startNode = Tree(self.NODEID, (i+1), self.TRACKID, parent=None, children=[], 
                             data=(self.MODEL.newState(detection, self.NEWLOG), t, t), 
                             weight=0, numMiss=0, endsTree=False)
            self.TRACKTREES[self.TRACKID] = TrackTree(startNode, self.NSCAN, t)
            self.TRACKID += 1

        # Print all track trees at the end of this iteration if in debug mode
        if self.DEBUG:
            print("All current track trees:")
            for tt in self.TRACKTREES:
                print(tt)
                self.TRACKTREES[tt].getRoot().printTree()
                print("\n")
            print("*"*100 + "\n")


    def iterate(self, detectLocs, t):
        """
        Move all the tracks forward to the given time and associate the given detections with
        existing tracks or create new tracks using a multiple hypothesis tracker

        detectLocs - A list of locations and covariances (x, P) corresponding to object detections
        t - The current time step (i.e. the time step at which the detections took place)
        """

        # Output printing at the start of every iteration
        print("Time           :", t)
        print("Num Track Trees:", len(self.TRACKTREES))
        print("Num Groups     :", len(self.GROUPS))
        print()


        if self.DEBUG:
            print("*"*100)
            print("Performing iteration with:")
            print("\tObject detections:", detectLocs)
            print("\tTime:", t, "\n")


        # Remove all tracks that should be ended from the possible end dictionary
        for tt in list(self.POSSIBLEEND.keys()):
            ended = self.POSSIBLEEND.pop(tt)
            if (ended.getNumUpdates() >= self.MINTRACKUPDATES):
                ended.lowerRoot()
                self.ENDEDTRACKS[tt] = ended
        self.POSSIBLEEND = {}

        # Get all possible object locations (all leaves in all track trees)
        leafNodes = []
        for tt in self.TRACKTREES:
            leafNodes += [(tt, leaf) for leaf in self.TRACKTREES[tt].getLeaves()]

        # For each leaf, find all possible detections that could match and create new branches
        missedCounter = -1
        detectionAssigned = [0]*len(detectLocs)
        for treeID, leaf in leafNodes:
            objectState, lastTime, lastUpdate = leaf.getData()
            objectState = self.MODEL.extrapolate(objectState, (t - lastTime))
            numMissedUpdates = leaf.getNumMissed()
            endLikelihood = max((1.0 - np.exp(-numMissedUpdates/self.ENDLAMBDA)), 1e-14)
            endLog = np.log(endLikelihood)
            contLog = np.log(1 - endLikelihood)

            if (not leaf.shouldEnd()):
                # For each object detection, see if it is close enough to the leaf node and create
                # a new branch/leaf in the case that it is
                for i, detection in enumerate(detectLocs):
                    if (self.MODEL.distance(detection, objectState) < self.GATE):
                        newState = self.MODEL.update(detection, objectState)
                        nodeWeight = (leaf.getWeight() + contLog + self.DETECTLOG + 
                                     self.MODEL.getScore(newState))
                        Tree(self.NODEID, (i+1), treeID, parent=leaf, children=[], data=(newState, t, t),
                             weight=nodeWeight, numMiss=0, endsTree=False)
                        detectionAssigned[i] = 1
                        self.NODEID += 1

                # Add instance where the object might be lost for a frame
                if ((t - lastUpdate) < self.MAXT):
                    nodeWeight = (leaf.getWeight() + contLog + self.SKIPLOG + 
                                 self.MODEL.getScore(objectState))
                    Tree(self.NODEID, 0, treeID, parent=leaf, children=[], 
                         data=(objectState, t, lastUpdate), weight=nodeWeight, 
                         numMiss=numMissedUpdates+1, endsTree=False)
                    missedCounter -= 1
                    self.NODEID += 1

                # Add instance where the track should end
                else:
                    nodeWeight = leaf.getWeight() + self.SKIPLOG + endLog
                    Tree(self.NODEID, 0, treeID, parent=leaf, children=[], 
                         data=(objectState, t, lastUpdate), weight=nodeWeight, 
                         numMiss=numMissedUpdates+1, endsTree=True)
                    self.NODEID += 1

            else:
                # Add instance where the tree should be ended
                Tree(self.NODEID, 0, treeID, parent=leaf, children=[], data=(objectState, t, lastUpdate),
                     weight=leaf.getWeight(), numMiss=numMissedUpdates+1, endsTree=True)
                self.NODEID += 1

        # Create a new root node for every object detection and make a note for shared detections
        for i, detection in enumerate(detectLocs):
            tempRoot = Tree(self.NODEID, 0, self.TRACKID, parent=None, children=[],
                            data=(detection + (0,), t, t), weight=0, numMiss=0, endsTree=False)
            self.NODEID += 1
            startNode = Tree(self.NODEID, (i+1), self.TRACKID, parent=tempRoot, children=[], 
                             data=(self.MODEL.newState(detection, self.NEWLOG), t, t), 
                             weight=self.NEWLOG, numMiss=0, endsTree=False)
            self.NODEID += 1
            falseAlarmNode = Tree(self.NODEID, -(i+1), treeID=self.TRACKID, parent=tempRoot, children=[], 
                                  data=(self.MODEL.newState(detection, self.FALSEALARM), t, t),
                                  weight=self.FALSEALARM, numMiss=1, endsTree=True)
            self.NODEID += 1
            falseNode = Tree(self.NODEID, 0, self.TRACKID, parent=tempRoot, children=[], 
                             data=(self.MODEL.newState(detection, 0), t, t), 
                             weight=0, numMiss=1, endsTree=True)
            self.NODEID += 1
            treeLeaves = [startNode, falseAlarmNode, falseNode]
            self.TRACKTREES[self.TRACKID] = TrackTree(tempRoot, self.NSCAN, t)

            # Add the newly created tree to its own group
            self.GROUPS.append(Group(self.MAXHYPOS, self.MINHYPRATIO, {self.TRACKID: treeLeaves}, 
                                     [[0, [tempRoot]]], self.NSCAN, self.MAXT, self.DEBUG))
            self.TRACKID += 1

        if self.DEBUG:
            print("Track trees after adding leaves:")
            for tree in self.TRACKTREES:
                self.TRACKTREES[tree].getLastPrune().printTree()
            print()

        # Assign group numbers to all of the track trees
        treeGroups = self.computeGroupNums(t)
        allGroupNums = {}
        for tree in treeGroups:
            allGroupNums[treeGroups[tree]] = set()

        # Split groups
        newGroupList = []
        for group in self.GROUPS:
            newGroupList += group.checkAndSplit(treeGroups)

        # Merge groups
        for i, group in enumerate(newGroupList):
            treesInGroup = group.getGroupTreeIDs()
            for tree in treesInGroup:
                allGroupNums[treeGroups[tree]].add(i)
        self.GROUPS = []
        for groupNum in allGroupNums:
            indices = allGroupNums[groupNum]
            group = newGroupList[indices.pop()]
            for index in indices:
                group.merge(newGroupList[index])
            if (not group.isEmpty()):
                self.GROUPS.append(group)
        
        # Prune and hypothesize for each group and end any trees that aren't chosen in the global
        # hypothesis
        possibleEndedTrees = []
        for group in self.GROUPS:
            endedTrees = group.generateGroupHypotheses(self.TRACKTREES, t)
            for tt in endedTrees:
                self.TRACKTREES[tt].end()
                ended = self.TRACKTREES.pop(tt)
                self.POSSIBLEEND[tt] = ended
                if (ended.getNumUpdates() >= self.NSCAN):
                    possibleEndedTrees.append([tt, ended])

        if self.DEBUG:
            print("All current track trees:")
            for tt in self.TRACKTREES:
                print(tt)
                self.TRACKTREES[tt].getRoot().printTree()
                print("\n")

        print("*"*100 + "\n")

        return possibleEndedTrees


    def computeGroupNums(self, timeNow):
        """
        Compute and return the group numbers for all current existing trees, which can then be used
        in assigning trees to groups, splitting and merging groups

        returns - A dictionary in which the keys are tree numbers and the values are their 
                  respective groups (i.e. trees with the same group number should be put in the 
                  same group)
        """

        # Set up lists to keep track of groups
        treeNums = list(self.TRACKTREES.keys())
        treeGroups = [-1]*len(treeNums)
        groupNum = 0

        # Go through all remaining track trees and note which of them have shared object detections
        toBeAdded = []
        detectHistory = []
        allSets = []
        for _ in range(self.NSCAN):
            detectHistory.append(dict())
        for tt in self.TRACKTREES:
            toBeAdded += self.TRACKTREES[tt].getLastPrune().getChildren()
            allSets.append({tt})
        while len(toBeAdded) > 0:
            nextNode = toBeAdded.pop()
            detectID = abs(nextNode.getDetectID())
            toBeAdded += nextNode.getChildren()
            if (detectID != 0):
                treeID = nextNode.getTreeID()
                t = nextNode.getData()[1]
                if (not detectID in detectHistory[timeNow - t]):
                    detectHistory[timeNow - t][detectID] = set()
                detectHistory[timeNow - t][detectID].add(treeID)

        # Get the sets for all shared detections
        for timeStamp in detectHistory:
            for detectID in timeStamp:
                allSets.append(timeStamp[detectID])

        # Combine all sets that have any shared detections
        out = []
        while len(allSets):
            first, *rest = allSets
            lf = -1
            while (len(first) > lf):
                lf = len(first)
                rest2 = []
                for r in rest:
                    if first.intersection(r):
                        first |= r
                    else:
                        rest2.append(r)
                rest = rest2
            out.append(first)
            allSets = rest

        # Assign numbers to each group
        returnDict = {}
        for groupNum, combined in enumerate(out):
            for treeID in combined:
                returnDict[treeID] = groupNum

        return returnDict


    def getGroups(self, timeNow):
        """
        Compute and return the groups for all current existing trees as a list

        returns - A list where every item in the list is a set of track tree IDs that belong
                  to the same group
        """

        # Set up lists to keep track of groups
        treeNums = list(self.TRACKTREES.keys())
        treeGroups = [-1]*len(treeNums)
        groupNum = 0

        # Go through all remaining track trees and note which of them have shared object detections
        toBeAdded = []
        detectHistory = []
        allSets = []
        for _ in range(self.NSCAN):
            detectHistory.append(dict())
        for tt in self.TRACKTREES:
            toBeAdded += self.TRACKTREES[tt].getLastPrune().getChildren()
            allSets.append({tt})
        while len(toBeAdded) > 0:
            nextNode = toBeAdded.pop()
            detectID = abs(nextNode.getDetectID())
            toBeAdded += nextNode.getChildren()
            if (detectID > 0):
                treeID = nextNode.getTreeID()
                t = nextNode.getData()[1]
                if (not detectID in detectHistory[timeNow - t]):
                    detectHistory[timeNow - t][detectID] = set()
                detectHistory[timeNow - t][detectID].add(treeID)

        # Get the sets for all shared detections
        for timeStamp in detectHistory:
            for detectID in timeStamp:
                allSets.append(timeStamp[detectID])

        # Combine all sets that have any shared detections
        out = []
        while len(allSets):
            first, *rest = allSets
            lf = -1
            while (len(first) > lf):
                lf = len(first)
                rest2 = []
                for r in rest:
                    if first.intersection(r):
                        first |= r
                    else:
                        rest2.append(r)
                rest = rest2
            out.append(first)
            allSets = rest

        return out


    def computeGlobalHypothesis(self, groups):
        """
        Computes the best global hypothesis for all track trees that have a depth greater or equal
        to the NSCAN parameter, used to prune trees.

        groups - A list of groups that have shared detections

        
        """

        globalHypothesis = []
        endedTrees = set()
        for group in groups:

            if self.DEBUG:
                print("\n" + "*"*100)
                print("Computing Global Hypothesis for group:", group)

            # Get all branches, save parent and identifier for each branch and compute branch weight, then prune
            allBranches = []
            branchNum = 0
            for tt in group:
                if (self.TRACKTREES[tt].getDepth() >= self.NSCAN):
                    rootList = self.TRACKTREES[tt].getBranches()
                    branchRoot = self.TRACKTREES[tt].getLastPrune()
                    for rootID, root in rootList:
                        branches = root.getFullBranches()
                        for branchNodes in branches:
                            allBranches.append([tt, [branchRoot, root] + branchNodes])
                            branchNum += 1

            if (len(allBranches) == 0):
                continue

            # Go through all track trees and compute the weight of each branch as well as noting shared 
            # detections for computing the graph
            branchWeights = {}
            sharedDetections = {}
            for i in range(self.NSCAN+1):
                sharedDetections[i] = {}
            for branchID, (trackTree, branch) in enumerate(allBranches):
                branchTotal = 0
                for i, node in enumerate(branch):
                    detectID = node.getDetectID()
                    if not (detectID in sharedDetections[i]):
                        sharedDetections[i][detectID] = set()
                    sharedDetections[i][detectID].add(branchID)
                    branchTotal += node.getWeight()
                branchWeights[branchID] = branchTotal

            # Debug printing
            if self.DEBUG:
                print("All branches that are considered for pruning")
                print("BranchNum BranchWeight - TrackTreeNumber: [BranchNodes]")
                for i, branch in enumerate(allBranches):
                    print(str(i) + " " + str(branchWeights[i]) + " - " + str(branch[0]) + ": " + str([b.getNodeID() for b in branch[1]]))
                print("*"*100)

            # Create weighted graph with edges indicating shared detections
            neighbours = {}
            for i in range(branchNum):
                neighbours[i] = set()
            for i in sharedDetections:
                for j in sharedDetections[i]:
                    shared = sharedDetections[i][j]
                    while shared:
                        one = shared.pop()
                        for two in shared:
                            neighbours[one].add(two)
                            neighbours[two].add(one)
            WG = WeightedGraph(graphDict=neighbours, weights=branchWeights, debug=self.DEBUG)

            # Debug print for created weighted graph
            if self.DEBUG:
                print("Neighbour list to be used for weighted graph")
                print(neighbours)
                print("*"*100)

            # Compute the best global hypothesis and return it
            mwis = WG.getMWIS()
            keptTrees = set()
            for keep in mwis:
                globalHypothesis.append([allBranches[keep][0], allBranches[keep][1][1].getNodeID()])
                keptTrees.add(allBranches[keep][0])
            for i in range(len(allBranches)):
                if (not allBranches[i][0] in keptTrees):
                    endedTrees.add(allBranches[i][0])

        # Debug print for kept branches
        if self.DEBUG:
            print("[Tree, Branch] to be kept:", globalHypothesis)
            print("Trees to be ended", endedTrees)
            print("*"*100 + "\n")

        return globalHypothesis, endedTrees


    def getEndedTracks(self):
        """
        Returns the dictionary containing track trees that have been ended
        """
        return self.ENDEDTRACKS


    def getCurrentTracks(self):
        """
        Returns the dictionary containing track trees that are currently active
        """

        return self.TRACKTREES


    def getAllTracks(self):
        """
        Returns the dictionaries containing the track trees that have been ended as well as track 
        trees that are still active
        """

        returnDict = self.ENDEDTRACKS.copy()
        for tt in self.TRACKTREES:
            if (self.TRACKTREES[tt].getNumUpdates() >= self.NSCAN):
                returnDict[tt] = self.TRACKTREES[tt]
        for tt in self.SUSPENDLIST:
            returnDict[tt] = self.SUSPENDLIST[tt][0]
        return returnDict




    def getAllObjectIDs(self):
        """
        Return a list of all of the object IDs that have been tracked
        """

        return(set(self.getAllTracks().keys()))
