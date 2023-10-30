#!/usr/bin/env python

###################################################################################################
#                                           Tree Class                                            #
#                                                                                                 #
# Class for creating Tree data structures that are able to keep track of both parents and         #
# children. These trees have been further updated to be used in tracking systems, so they have    #
# functions and variables to keep track of detection numbers, weights as well as hold additional  #
# data for the state of the tracked object.                                                       #
#                                                                                                 #
# Author: Marco Grond                                                                             #
# Version: 0.1.0                                                                                  #
###################################################################################################

from itertools import chain, repeat, starmap
from operator import add
import numpy as np

class Tree:

    def __init__(self, nodeID, detectionID, treeID, parent=None, children=[], data=None, weight=0, 
                 numMiss=0, endsTree=False):
        """
        Create a tree node

        nodeID - The name or id for the given node
        detectionID - The detection that this tree node is associated with
        treeID - The ID of the tree that this node belongs to
        parent - A tree node that denotes the parent of this node. If none is provided, this node
                 is assumed to be a root node
        children - A list of child nodes, of the type Tree
        data - Any type of data that should be stored in the tree
        weight - The weight for this node in the track tree
        numMiss - The number of detections that have been missed up to this point
        endsTree - A flag to indicate whether or not this node ends the tree
        """

        self.TREEID = treeID
        self.NODEID = nodeID
        self.DETECTID = detectionID
        self.DATA = data
        self.WEIGHT = weight
        self.DEPTH = 0
        self.CHILDREN = {}
        self.PARENT = None
        self.NUMMISSED = max(0, numMiss)
        self.ENDSTREE = endsTree

        # Set the parent of this node, if it is provided
        if isinstance(parent, Tree):
            self.PARENT = parent
            parent.addChild(self)

        # Set the children of this node, if any are provided
        if not (children is None):
            for c in children:
                if isinstance(c, Tree):
                    self.CHILDREN[c.getNodeID()] = c
                    c.setParent(self)


    def addChild(self, childNode):
        """
        Adds a child node, of type Tree, to the root node

        childNode - A node of type Tree to be added as a child
        """

        if isinstance(childNode, Tree):
            self.CHILDREN[childNode.getNodeID()] = childNode
            childNode.setDepth(self.DEPTH + 1)
            if ((childNode.getParent() is None) or 
                    (self.NODEID != childNode.getParent().getNodeID())):
                childNode.setParent(self)


    def removeChild(self, childIdentifier):
        """
        Removes and returns the child node with the given identifier if it is a direct descendent 
        of this node in the tree

        childIdentifier - The identifier for the child that should be removed
        """

        if (childIdentifier in self.CHILDREN):
            return self.CHILDREN.pop(childIdentifier)
        return None


    def removeAllChildren(self):
        """
        Removes all children from this node and set their parents to be None
        """

        for child in self.CHILDREN:
            self.CHILDREN[child].setParent(None)
        self.CHILDREN = {}


    def removeThisLeaf(self):
        """
        Remove this leaf from the tree, as well its ancesteral leaves if they don't have any other
        children
        """

        if (not self.PARENT is None):
            self.PARENT.removeChild(self.NODEID)
            if (len(self.PARENT.getChildrenDict()) <= 0):
                self.PARENT.removeThisLeaf()
            self.PARENT = None


    def setParent(self, parentNode):
        """
        Set the parent of this node

        parentNode - A Tree node which should be the parent of the current node
        """

        if isinstance(parentNode, Tree):
            self.PARENT = parentNode
            self.DEPTH = parentNode.getDepth() + 1
            if not (self.NODEID in parentNode.getChildrenDict()):
                self.PARENT.addChild(self)
        elif (parentNode is None):
            self.PARENT = None


    def setDepth(self, newDepth):
        """
        Set the depth of this node. Only for use by other functions in this class

        newDepth - The new depth that this node is at
        """

        self.DEPTH = newDepth
        for c in self.CHILDREN:
            self.CHILDREN[c].setDepth(newDepth + 1)


    def setWeight(self, weight):
        """
        Sets the weight for this node
        """

        self.WEIGHT = weight


    def getChildrenDict(self):
        """
        Returns a dictionary in which the keys are identifiers for child nodes and the values 
        the associated child node
        """

        return self.CHILDREN


    def getChildren(self):
        """
        Returns a list of all child nodes for the current node
        """

        return [self.CHILDREN[i] for i in self.CHILDREN]


    def numChildren(self):
        """
        Returns the number of children that this node has
        """

        return len(self.CHILDREN)


    def getChild(self, childID):
        """
        Return the node for the child with the given ID
        """

        if (childID in self.CHILDREN):
            return self.CHILDREN[childID]
        return None


    def getChildDetectID(self, detectID):
        """
        Returns the node for the child that has the given detection ID
        """

        for childID in self.CHILDREN:
            if (self.CHILDREN[childID].getDetectID() == detectID):
                return self.CHILDREN[childID]
        return None


    def getParent(self):
        """
        Returns the parent of the current node
        """

        return self.PARENT


    def getWeight(self):
        """
        Returns the weight for this node
        """

        return self.WEIGHT


    def getNumMissed(self):
        """
        Get the number of detections that have been missed up to this point
        """

        return self.NUMMISSED


    def getTreeID(self):
        """
        Returns the ID of the tree that this node belongs to
        """

        return self.TREEID


    def setTreeID(self, newID):
        """
        Set the tree ID for this tree to the new ID

        newID - The new tree ID (as an integer) used to identify nodes in this tree
        """

        self.TREEID = newID


    def changeTreeID(self, newID):
        """
        Recursively change the tree ID to the newly given ID
        """

        # Change the tree ID
        if (self.TREEID == newID):
            return
        self.TREEID = newID

        # Change the parent tree ID
        if (self.PARENT is None):
            return
        self.PARENT.changeTreeID(newID)


    def getNodeID(self):
        """
        Returns the ID of the current node in the tree
        """

        return self.NODEID


    def getDetectID(self):
        """
        Returns the unique number (additional identifier) assigned to this tree node
        """

        return self.DETECTID


    def getDepth(self):
        """
        Returns the depth of this node in the tree
        """

        return self.DEPTH


    def shouldEnd(self):
        """
        Returns the flag to indicate whether or not a node ends a tree
        """

        return self.ENDSTREE


    def toList(self):
        """
        Returns a list representation of the tree with the current node as the root node
        """

        returnList = [self.NODEID]
        for c in self.CHILDREN:
            returnList.append(self.CHILDREN[c].toList())
        return returnList


    def setData(self, data):
        """
        Set the data that should be stored in this node

        data - Any type of data that should be stored
        """

        self.DATA = data


    def getData(self):
        """
        Returns the data that is stored in this node
        """

        return self.DATA


    def getFullBranches(self):
        """
        Returns a list of lists containing all tree nodes in all branches from the current node
        """

        # Return condition for when a node has no children
        if (len(self.CHILDREN) == 0):
            return [[]]

        # Get all branches for each child
        branchList = []
        for c in self.CHILDREN:
            childBranches = self.CHILDREN[c].getFullBranches()
            for b in childBranches:
                branchList.append([self.CHILDREN[c]] + b)

        return branchList


    def getLeaves(self):
        """
        Returns a list containing all of the leaves of the current node
        """

        # Set up and check if current node is a leaf
        childList = self.getChildren()
        if (len(childList) == 0):
            return [self]
        leafList = []

        # Traverse the entire tree and keep record of all leaves
        while (len(childList) > 0):
            currentChild = childList.pop(0)
            newChildren = currentChild.getChildren()
            if (len(newChildren) == 0):
                leafList.append(currentChild)
            else:
                childList += newChildren

        return leafList


    def getLeafIDs(self):
        """
        Returns a list of the IDs of leaf nodes for the current node
        """

        leaves = self.getLeaves()
        returnList = []
        for leaf in leaves:
            returnList.append(leaf.getNodeID())
        return returnList


    def printTree(self, includeData=True):
        """
        Print the tree starting at the current node

        includeData - A flag to visualize the data along with the name of the nodes. If False, only
                      the node ids will be used in the visualization.
        """

        print("Tree ID:", self.TREEID)
        print(self.printHelper(includeData=includeData, indent="", final=False, siblings=False))


    def printHelper(self, includeData=True, indent="", final=False, siblings=False):
        """
        Recursive helper function to get a string visualization of the tree
        """

        if (len(indent) > 0):
            preString = indent + ("└─ " if (final and len(indent) > 0) else "├─ ")
        else:
            preString = indent

        numChildren = len(self.CHILDREN)
        if includeData:
            retString = ((preString + "(%d) %d - %.5f - ((" + str(list(self.DATA[0][0].flatten()[:2])) + 
                         ", " + str(self.DATA[0][2]) + "), " + str(self.DATA[1]) + ", " + 
                         str(self.DATA[2]) + ")\n") % (self.DETECTID, self.NODEID, self.WEIGHT))
        else:
            retString = (preString + "(%d) %d - %.5f\n" % (self.DETECTID, self.NODEID, self.WEIGHT))
        for i, c in enumerate(self.CHILDREN):
            newIndent = indent + ("│  " if siblings else "   ")
            retString += self.CHILDREN[c].printHelper(includeData,
                                                      newIndent, 
                                                      (i == (numChildren-1)), 
                                                      ((numChildren>1) and (i<(numChildren-1))))
        return retString


    def __repr__(self):
        """
        To string method for tree object
        """

        return ("(" + str(self.NODEID) + " (" + str(self.DETECTID) + ") -> (" + 
                str(list(self.DATA[0][0].flatten()[:2])) + ", " + str(self.DATA[0][2]) + "), " + 
                str(self.DATA[1]) + ", " + str(self.DATA[2]) + ")")


    def __eq__(self, other):
        """
        Override of the equality operator to test if two nodes are equal
        """

        return (self.getDetectID() == other.getDetectID())


    def __ne__(self, other):
        """
        Override of the not-equal operator to test if two nodes are not-equal
        """

        return (self.getDetectID() != other.getDetectID())


    def __lt__(self, other):
        """
        Override of the less than operator to test if this node is less than the other node
        """

        return (self.getDetectID() < other.getDetectID())
        #if (self.getData().getScore() < other.getData().getScore()):
        #    return True
        #elif (self.getData().getScore() == other.getData().getScore()):
        #    if (self.getTreeID() < other.getTreeID()):
        #        return True
        #    elif (self.getTreeID() == other.getTreeID()):
        #        if (self.getNodeID() < other.getNodeID()):
        #            return True
        #return False


    def __le__(self, other):
        """
        Override of the less than or equal operator to test if this node is less than or equal to 
        the other node
        """

        return (self.getDetectID() <= other.getDetectID())


    def __gt__(self, other):
        """
        Override of the greater than operator to test if this node is greater than the other node
        """

        return (self.getDetectID() > other.getDetectID())
        #if (self.getData().getScore() > other.getData().getScore()):
        #    return True
        #elif (self.getData().getScore() == other.getData().getScore()):
        #    if (self.getTreeID() > other.getTreeID()):
        #        return True
        #    elif (self.getTreeID() == other.getTreeID()):
        #        if (self.getNodeID() > other.getNodeID()):
        #            return True
        #return False


    def __ge__(self, other):
        """
        Override of the greater than or equal operator to test if this node is greater than or 
        equal to the other node
        """

        return (self.getDetectID() >= other.getDetectID())



if __name__ == '__main__':
    root = Tree(identifier='a', parent=None, children=[], data=1)
    n11 = Tree(identifier='f', parent=None, children=[], data=6)
    n12 = Tree(identifier='g', parent=None, children=[], data=7)
    n13 = Tree(identifier='h', parent=None, children=[], data=8)
    n111 = Tree(identifier='i', parent=n11, children=[], data=9)
    n112 = Tree(identifier='j', parent=n11, children=[], data=10)
    n1111 = Tree(identifier='k', parent=n111, children=[], data=11)
    n121 = Tree(identifier='l', parent=n12, children=[], data=12)
    n121 = Tree(identifier='z', parent=n12, children=[], data=12)
    n1 = Tree(identifier='b', parent=None, children=[n11, n12, n13], data=2)
    n2 = Tree(identifier='c', parent=root, children=[], data=3)
    n4 = Tree(identifier='e', parent=root, children=[], data=5)
    n3 = Tree(identifier='d', parent=None, children=[], data=4)
    n31 = Tree(identifier='m', parent=n3, children=[], data=13)
    n311 = Tree(identifier='n', parent=n31, children=[], data=14)
    n3111 = Tree(identifier='o', parent=n311, children=[], data=15)
    n31111 = Tree(identifier='p', parent=n3111, children=[], data=16)
    n311111 = Tree(identifier='q', parent=n31111, children=[], data=17)
    n41 = Tree(identifier='r', parent=n4, children=[], data=18)
    n42 = Tree(identifier='s', parent=n4, children=[], data=19)
    n411 = Tree(identifier='t', parent=n41, children=[], data=20)
    n412 = Tree(identifier='u', parent=n41, children=[], data=21)
    n421 = Tree(identifier='v', parent=n42, children=[], data=22)
    n422 = Tree(identifier='w', parent=n42, children=[], data=23)
    n4221 = Tree(identifier='x', parent=n422, children=[], data=24)
    n1.setParent(root)
    root.addChild(n3)

    root.printTree()
    print()
    n4.printTree()

    rLeaves = root.getLeaves()
    for l in rLeaves:
        print(l)

    print(root.getFullBranches())
