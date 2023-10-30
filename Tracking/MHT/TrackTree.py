#!/usr/bin/env python

###################################################################################################
#                                        Track Tree Class                                         #
#                                                                                                 #
# Class for creating and maintaining Track Trees used to keep record of tracks and possible data  #
# associations that can be made, while allowing pruning of branches that get too long.            #
#                                                                                                 #
# Author: Marco Grond                                                                             #
# Version: 0.1.0                                                                                  #
###################################################################################################

from Tree import Tree

class TrackTree():

    def __init__(self, root, pruneDepth, time):
        """
        Create a node for the track tree.

        root - The root node for the track tree
        pruneDepth - The depth that a branch is allowed to be before pruning should take place
        time - The time at which the root node was created

        identifier - The name or id for the given node
        state - The state of the track at this node (i.e. an instance of the Kalman state)
        parent - A tree node that denotes the parent of this node. If none is provided, this node
                 is assumed to be a root node
        children - A list of child nodes, of the type Tree
        """


        self.ROOTNODE = root
        self.LASTPRUNE = None
        self.PRUNEDEPTH = pruneDepth
        self.NUMUPDATES = 0
        self.LASTUPDATE = time


    def getRoot(self):
        """
        Returns the root node for this tree
        """

        return self.ROOTNODE


    def setRoot(self, newRoot):
        """
        Update the root node of this tree to the given node

        newRoot - The new root node for this track tree, of type Tree
        """

        self.ROOTNODE = newRoot


    def lowerRoot(self):
        """
        Moves the root of this tree to the child of the root, if there is only a single child
        """

        children = self.ROOTNODE.getChildrenDict()
        if (len(children) == 1):
            child = next(iter(children.values()))
            self.ROOTNODE.removeAllChildren()
            self.ROOTNODE = child


    def getBranches(self):
        """
        Returns a list of branches that need to be pruned. The returned list contains tuples where
        the first entry is the branch ID and the second entry is a root node for that branch.

        returns - [(id1, node1), (id2, node2), ...]
        """

        # Only get children for deepest node that can possibly have more than one child
        if (self.LASTPRUNE is None):
            children = self.ROOTNODE.getChildrenDict()
        else:
            children = self.LASTPRUNE.getChildrenDict()
        return [(i, children[i]) for i in children]


    def getLeaves(self):
        """
        Returns a list of all of the leaf nodes for this track tree
        """

        # Only find leaves for the deepest node that can possibly have more than one child
        lowestNode = self.LASTPRUNE
        if lowestNode is None:
            lowestNode = self.ROOTNODE
        return lowestNode.getLeaves()


    def end(self):
        """
        Removes all brances stemming from the most recently pruned node.
        """     

        # Find the node that was pruned most recently
        if (self.LASTPRUNE is None):
            pruneNode = self.ROOTNODE
        else:
            pruneNode = self.LASTPRUNE

        # Prune all child nodes and set the chosen child node as the new most recently pruned node
        children = list(pruneNode.getChildrenDict().keys())
        for i in children:
            pruneNode.removeChild(i)


    def keep(self, chosenBranch, time):
        """
        Prune all branches, except for the provided chosen branch, from the node that was pruned
        most recently. Returns whether or not the track tree has ended.

        chosenBranch - The ID of the branch that should be kept, while the rest are pruned
        time - The time at which this branch was chosen

        returns - True if the chosenBranch exists and nodes were pruned and False if it does not
                  exist and no nodes were pruned, as well as a list of the pruned leaf IDs
        """

        # Find the node that was pruned most recently
        if (self.LASTPRUNE is None):
            pruneNode = self.ROOTNODE
        else:
            pruneNode = self.LASTPRUNE

        # Prune all child nodes and set the chosen child node as the new most recently pruned node
        children = pruneNode.getChildrenDict()
        childKeys = list(children.keys())
        allRemovedLeaves = []
        if not chosenBranch in children:
            return False, []
        for i in childKeys:
            if not (i == chosenBranch):
                child = pruneNode.removeChild(i)
                allRemovedLeaves += child.getLeafIDs()
            else:
                self.LASTPRUNE = children[i]

        # Update the last time that an actual detection was associated with this track, if the 
        # chosen branch id is positive
        if (self.LASTPRUNE.getDetectID() > 0):
            self.LASTUPDATE = self.LASTPRUNE.getData()[2]
            self.NUMUPDATES += 1

        return True, allRemovedLeaves


    def mergeTrees(self, newTree):
        """
        Merge another track tree with this tree. This operation is only allowed when this track
        tree has a single leaf node.

        newTree - A new track tree that should be added onto this track tree
        """

        # Check to make sure that this track tree only has a single leaf
        leaf = self.getLeaves()
        if (len(leaf) != 1):
            print("Merged track tree", self.ROOTNODE.getTreeID(), "has more than one leaf:")
            self.ROOTNODE.printTree()
            exit()

        # Update the treeIDs for this tree
        leaf = leaf[0]
        currentNodes = [self.getRoot()]
        newID = newTree.getRoot().getTreeID()
        while (len(currentNodes) > 0):
            cn = currentNodes.pop()
            cn.setTreeID(newID)
            currentNodes += cn.getChildren()
        #leaf.changeTreeID(newTree.getRoot().getTreeID())
        #newLeaves = newTree.getLeaves()
        #for l in newLeaves:
        #    l.changeTreeID(leaf.getTreeID())

        # Set the child for the leaf node to be the root node of newTree
        newTree.lowerRoot()
        leaf.addChild(newTree.getRoot())

        # Update additional Track Tree variables
        #self.LASTPRUNE = newTree.getLastPrune()
        #self.NUMUPDATES += newTree.getNumUpdates()
        newTree.setRoot(self.ROOTNODE)
        newTree.setNumUpdates(newTree.getNumUpdates() + self.NUMUPDATES)


    def getDepth(self):
        """
        Get the depth of this track tree
        """

        leafList = self.getLeaves()
        if (len(leafList) == 0):
            return -1
        depth = leafList[0].getDepth()
        for l in leafList[1:]:
            if (l.getDepth() > depth):
                depth = l.getDepth()
        return (depth+1)


    def getLastPrune(self):
        """
        Returns the last pruned node. If the tree has not been pruned, the last pruned node is the 
        root
        """

        if (self.LASTPRUNE is None):
            return self.ROOTNODE
        return self.LASTPRUNE


    def getLastUpdate(self):
        """
        Returns when last this track tree was updated with a detection.
        """

        return self.LASTUPDATE


    def getNumUpdates(self):
        """
        Returns the number of times data association has taken place with a real detection
        """

        return self.NUMUPDATES


    def setNumUpdates(self, updates):
        """
        Sets the number of updates that this tree has received to the given value

        updates - The new number of updates for this tree
        """

        self.NUMUPDATES = updates


    def getLocations(self, model):
        """
        Returns a dictionary of all of the locations for the tracked item, with the times as keys

        model - A model that can be used to extract the location from the data
        """

        returnList = {}
        node = self.ROOTNODE.getChildren()[0]
        returnList[node.getData()[1]] = model.getLocation(node.getData()[0])
        while (node.numChildren() >= 1):
            node = node.getChildren()[0]
            returnList[node.getData()[1]] = model.getLocation(node.getData()[0])
        return returnList


    def getLocationsList(self, model):
        """
        Returns a list of all of the tracked times, locations and whether the detection was true
        or not, in the format [(t1, (x1, y1), T1), (t2, (x2, y2) T2), ...]

        model - A model that can be used to extract the location from the data
        """

        returnList = []
        node = self.ROOTNODE#.getChildren()[0]
        x, y = model.getLocation(node.getData()[0])
        returnList.append((node.getData()[1], (float(x), float(y)), (node.getDetectID() > 0)))
        while (node.numChildren() >= 1):
            node = node.getChildren()[0]
            x, y = model.getLocation(node.getData()[0])
            returnList.append((node.getData()[1], (float(x), float(y)), (node.getDetectID() > 0)))
        return returnList



if __name__ == '__main__':
    root = Tree(0, parent=None, data=0)
    tt = TrackTree(root, 3, 0)
    c1 = Tree(1, parent=root, data=1)
    c2 = Tree(2, parent=c1, data=2)
    c3 = Tree(3, parent=c2, data=3)
    c4 = Tree(4, parent=c2, data=4)
    c5 = Tree(5, parent=c2, data=5)
    c20 = Tree(6, parent=c2, data=6)
    c6 = Tree(7, parent=c3, data=3)
    c7 = Tree(8, parent=c3, data=3)
    c8 = Tree(9, parent=c3, data=3)
    c9 = Tree(10, parent=c4, data=4)
    c10 = Tree(11, parent=c4, data=4)
    c11 = Tree(12, parent=c5, data=5)
    c12 = Tree(13, parent=c5, data=5)

    for c in tt.getLeaves():
        print(c)
    print(tt.getBranches())
    root.printTree()
    print(tt.keep(1, 1), '\n')
    root.printTree()
    print(tt.keep(1, 2), '\n')
    root.printTree()
    print(tt.keep(2, 3), '\n')
    root.printTree()
    print(tt.keep(7, 4), '\n')
    root.printTree()
    print(tt.keep(4, 5), '\n')
    root.printTree()
