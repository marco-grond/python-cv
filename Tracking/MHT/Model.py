#!/usr/bin/env python

###################################################################################################
#                                      Kalman filter class                                        #
#                                                                                                 #
# Class consisting of a normal Kalman filter as well as a class to manage multiple Kalman filters #
# Additional functionality is added to the Kalman filter, such as the ability to keep track of    #
# all past states, suspend a track and restart a track. Additional functions, such as computing   #
# the Mahalanobis distance and a score indicating how well a detection matches the track are also #
# provided.                                                                                       #
#                                                                                                 #
# Author: Marco Grond                                                                             #
# Version: 0.1.0                                                                                  #
###################################################################################################

import numpy as np
import math
from collections import deque
from copy import deepcopy
from scipy.stats import multivariate_normal as normal


class BoundingBoxKalman:
    """
    Class for the basic operations of a Kalman filter implemented with bounding boxes. It does not 
    keep track of the state of the object, but just applies the Kalman operation to the provided 
    state.

    Measured state of an object is given as: [x, y, w, h]
    Object state is given as: [x, y, w, h, dx, dy, dw, dh]
    """

    def __init__(self, logNormFactor=1.5963597, qDist=1, qSize=1, qVel=1, qGrowth=1, r=8):
        """
        logNormFactor - A factor used when computing the score for a state and measurement (uses 
                        distance along with this factor to compute the score)
        qDist - The location error introduced in the model, used to create the first two rows of 
                the Q matrix (assumed to be the same for x and y)
        qSize - The size error for the bounding box introduced in the model, used in the creation
                of the Q matrix (assumed to be the same for h and w)
        qVel - The velocity error introduced in the model, used to create the last two rows of the
               Q matrix (assumed to be the same for x and y)
        qGrowth - The error introduced for the changing of the size of the bounding box introduced
                  in the model, used in the creation of the Q matrix (assumed to be the same for 
                  h and w)
        r - The measurement error used to create the R matrix (assumed to be the same for 
            x, y, w and h)
        """

        # Initialize arrays used for the Kalman filter
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0]])
        self.Q = np.array([[qDist, 0, 0, 0, 0, 0, 0, 0],
                           [0, qDist, 0, 0, 0, 0, 0, 0],
                           [0, 0, qSize, 0, 0, 0, 0, 0],
                           [0, 0, 0, qSize, 0, 0, 0, 0],
                           [0, 0, 0, 0, qVel, 0, 0, 0,],
                           [0, 0, 0, 0, 0, qVel, 0, 0,],
                           [0, 0, 0, 0, 0, 0, qGrowth, 0],
                           [0, 0, 0, 0, 0, 0, 0, qGrowth]])
        self.R = np.identity(4) * r

        # Initialize variables used for computing scores
        self.LOGNORM = logNormFactor


    def extrapolate(self, kalmanState, timeStep=1):
        """
        Project the object forward in time

        kalmanState - The current state of the object in the form (x, P, score, history)
        timeStep - The time interval between the last extrapolation and this extrapolation

        returns - The state of the projected object, in the form (x_, P_, score_)
        """

        stateLoc, stateCov, _, stateHist = kalmanState
        stateHist = stateHist.copy()
        stateHist.append(stateLoc)

        # Move the point forward by the given time step and update the covariance and score
        F = np.array([[1, 0, 0, 0, timeStep, 0, 0, 0],
                      [0, 1, 0, 0, 0, timeStep, 0, 0],
                      [0, 0, 1, 0, 0, 0, timeStep, 0],
                      [0, 0, 0, 1, 0, 0, 0, timeStep],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]])
        x_ = np.matmul(F, stateLoc)
        P_ = np.matmul(np.matmul(F, stateCov), F.transpose()) + self.Q

        return (x_, P_, 0, stateHist)


    def update(self, measurement, kalmanState):
        """
        Update the object's projected location with a measurement

        measurement - The state of the object after extrapolation in the form (location, covariance)
        kalmanState - The current state of the object in the form (x_, P_, score_, history)

        returns - The new state of the object, in the form (x, P, score)
        """

        # Unpack the variables from the object state and measured state
        z, P = measurement
        x_, P_, score_, hist = kalmanState

        # Calculate innovation and innovation covariance
        inno = z - np.matmul(self.H, x_)
        S = np.matmul(np.matmul(self.H, P_), self.H.transpose()) + self.R

        # Update position and covariance
        K = np.matmul(np.matmul(P_, self.H.transpose()), np.linalg.inv(S))*1
        x = x_ + np.matmul(K, inno)
        P = np.matmul((np.eye(8) - np.matmul(K, self.H)), P_)

        # Get the motion score for this object
        score = -(self.LOGNORM + np.log(np.linalg.det(S))/2 +
                 np.matmul(np.matmul(np.transpose(inno), np.linalg.inv(S)), inno)/2)

        return (x, P, np.asscalar(score), hist)


    def getMotionScore(self, measurement, objectState):
        """
        Returns the motion score between the current Kalman state estimate and the given 
        measurement, using a normal distribution

	measurement - The measured object state (z, P) as a numpy array
        objectState - The current state of the object in the form (x, P, score, history)

        returns - The likelihood that this object is the same object as the tracked object 
                  represented by the Kalman state
        """

        return np.log(normal.pdf(x=measurement[0].flatten()[:2], mean=objectState[0].flatten()[:2],
                                 cov=objectState[1][0:2, 0:2]))


    def distance(self, measurement, kalmanState):
        '''
        Compute the Mahalanobis distance between a measured location and the state of a Kalman 
        object

        measurement - The measured object location and coviariance in the form (z, P)
        kalmanState - The state of the tracked object, in the form (x, P, score)
        '''

        z, P = measurement
        x_, P_, score, hist = kalmanState
        innov = z - np.matmul(self.H, x_)
        innovCov = np.matmul(np.matmul(self.H, P_), self.H.transpose()) + self.R
        dist = np.matmul(np.matmul(np.transpose(innov), np.linalg.inv(innovCov)), innov)
        return dist[0, 0]


    def getLocation(self, objectState):
        """
        Returns the location from the current state of the object

        objectState - The current state of the object in the form (x, P, score, history)
        """

        return objectState[0].flatten()[:2]


    def getCovariance(self, objectState):
        """
        Returns the covariance matrix from the current state of the object

        objectState - The current state of the object in the form (x, P, score, history)
        """

        return objectState[1]


    def getScore(self, objectState):
        """
        Returns the score assigned to the current state of the object

        objectState - The current state of the object in the form (x, P, score, history)
        """

        return objectState[2]


    def getRestartScore(self, suspendedObject, newObject):
        """
        Returns the score associated with restarting the suspended track with the newly detected
        object

        suspendedObject - The state of the suspended object which might be restarted
        newObject - The state of the newly detected object
        """

        return np.log(1/self.distance(newObject, suspendedObject))


    def newState(self, measurement, startingWeight):
        """
        Creates a new state from the measurement with the given starting weight

        measurement - A measurement of a possible object detection in the form (z, P)
        startingWeight - The starting weight that should be assigned to new objects
        """

        pos, Cov = measurement
        stateLoc = np.ones((8, 1))
        stateLoc[0:4, :] = pos[0:4, :]
        stateHistory = deque(maxlen=20)
        stateHistory.append(stateLoc)
        return (stateLoc, Cov, startingWeight, stateHistory)


    def getStateHist(self, objectState):
        """
        Returns the state history for the previous 20 time steps
        """

        return objectState[3]


    def getAvgVelocity(self, objectState, beginIndex=0, endIndex=0):
        """
        Get the average velocity for the object state using the state history, only taking 
        instances into account that fall within the begin and end indices

        objectState - The current state of the object in the form (x, P, score, history)
        beginIndex - How many of the first state entries should be ignored (0 means all are used)
        endIndex - How many of the last state entries should be ignored (0 means all are used)
        """

        # Add all of the state histories within the given indices together
        beginIndex = max(beginIndex, 0)
        endIndex = max(len(objectState[3])-endIndex, 0)
        if (endIndex <= beginIndex):
            return np.zeros((2, 1))
        avgState = np.zeros((4, 1))
        for i in range(beginIndex, endIndex):
            avgState += objectState[3][i]

        # Compute and return the average velocity
        avgVelocity = (avgState/(endIndex - beginIndex))[2:]
        return avgVelocity



class BaseKalman:
    """
    Class for the basic operations of a Kalman filter. It does not keep track of the state of the
    object, but just applies the Kalman operation to the provided state
    """

    def __init__(self, logNormFactor=1.5963597, qDist=1, qVel=1, r=8):
        """
        logNormFactor - A factor used when computing the score for a state and measurement (uses 
                        distance along with this factor to compute the score)
        qDist - The location error introduced in the model, used to create the first two rows of 
                the Q matrix (assumed to be the same for x and y)
        qVel - The velocity error introduced in the model, used to create the last two rows of the
               Q matrix (assumed to be the same for x and y)
        r - The measurement error used to create the R matrix (assumed to be the same for x and y)
        """

        # Initialize arrays used for the Kalman filter
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.array([[qDist, 0, 0, 0],
                           [0, qDist, 0, 0],
                           [0, 0, qVel, 0],
                           [0, 0, 0, qVel]])
        self.R = np.identity(2) * r

        # Initialize variables used for computing scores
        self.LOGNORM = logNormFactor


    def extrapolate(self, kalmanState, timeStep=1):
        """
        Project the object forward in time

        kalmanState - The current state of the object in the form (x, P, score, history)
        timeStep - The time interval between the last extrapolation and this extrapolation

        returns - The state of the projected object, in the form (x_, P_, score_)
        """

        stateLoc, stateCov, _, stateHist = kalmanState
        stateHist = stateHist.copy()
        stateHist.append(stateLoc)

        # Move the point forward by the given time step and update the covariance and score
        F = np.array([[1, 0, timeStep, 0],
                      [0, 1, 0, timeStep],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        x_ = np.matmul(F, stateLoc)
        P_ = np.matmul(np.matmul(F, stateCov), F.transpose()) + self.Q

        return (x_, P_, 0, stateHist)


    def update(self, measurement, kalmanState):
        """
        Update the object's projected location with a measurement

        measurement - The state of the object after extrapolation in the form (location, covariance)
        kalmanState - The current state of the object in the form (x_, P_, score_, history)

        returns - The new state of the object, in the form (x, P, score)
        """

        # Unpack the variables from the object state and measured state
        z, P = measurement
        x_, P_, score_, hist = kalmanState

        # Calculate innovation and innovation covariance
        inno = z - np.matmul(self.H, x_)
        S = np.matmul(np.matmul(self.H, P_), self.H.transpose()) + self.R

        # Update position and covariance
        K = np.matmul(np.matmul(P_, self.H.transpose()), np.linalg.inv(S))*1
        x = x_ + np.matmul(K, inno)
        P = np.matmul((np.eye(4) - np.matmul(K, self.H)), P_)

        # Get the motion score for this object
        score = -(self.LOGNORM + np.log(np.linalg.det(S))/2 +
                 np.matmul(np.matmul(np.transpose(inno), np.linalg.inv(S)), inno)/2)

        return (x, P, np.asscalar(score), hist)


    def getMotionScore(self, measurement, objectState):
        """
        Returns the motion score between the current Kalman state estimate and the given 
        measurement, using a normal distribution

	    measurement - The measured object state (z, P) as a numpy array
        objectState - The current state of the object in the form (x, P, score, history)

        returns - The likelihood that this object is the same object as the tracked object 
                  represented by the Kalman state
        """

        return np.log(normal.pdf(x=measurement[0].flatten()[:2], mean=objectState[0].flatten()[:2],
                                 cov=objectState[1][0:2, 0:2]))


    def distance(self, measurement, kalmanState):
        '''
        Compute the Mahalanobis distance between a measured location and the state of a Kalman 
        object

        measurement - The measured object location and coviariance in the form (z, P)
        kalmanState - The state of the tracked object, in the form (x, P, score)
        '''

        z, P = measurement
        x_, P_, score, hist = kalmanState
        innov = z - np.matmul(self.H, x_)
        innovCov = np.matmul(np.matmul(self.H, P_), self.H.transpose()) + self.R
        dist = np.matmul(np.matmul(np.transpose(innov), np.linalg.inv(innovCov)), innov)
        return dist[0, 0]


    def getLocation(self, objectState):
        """
        Returns the location from the current state of the object

        objectState - The current state of the object in the form (x, P, score, history)
        """

        return objectState[0].flatten()[:2]


    def getCovariance(self, objectState):
        """
        Returns the covariance matrix from the current state of the object

        objectState - The current state of the object in the form (x, P, score, history)
        """

        return objectState[1]


    def getScore(self, objectState):
        """
        Returns the score assigned to the current state of the object

        objectState - The current state of the object in the form (x, P, score, history)
        """

        return objectState[2]


    def getRestartScore(self, suspendedObject, newObject):
        """
        Returns the score associated with restarting the suspended track with the newly detected
        object

        suspendedObject - The state of the suspended object which might be restarted
        newObject - The state of the newly detected object
        """

        x_, P_, score, visual_features = suspendedObject
        return score


    def newState(self, measurement, startingWeight):
        """
        Creates a new state from the measurement with the given starting weight

        measurement - A measurement of a possible object detection in the form (z, P)
        startingWeight - The starting weight that should be assigned to new objects
        """

        pos, Cov = measurement
        stateLoc = np.ones((4, 1))
        stateLoc[0:2, :] = pos[0:2, :]
        stateHistory = deque(maxlen=20)
        stateHistory.append(stateLoc)
        return (stateLoc, Cov, startingWeight, stateHistory)


    def getStateHist(self, objectState):
        """
        Returns the state history for the previous 20 time steps
        """

        return objectState[3]


    def getAvgVelocity(self, objectState, beginIndex=0, endIndex=0):
        """
        Get the average velocity for the object state using the state history, only taking 
        instances into account that fall within the begin and end indices

        objectState - The current state of the object in the form (x, P, score, history)
        beginIndex - How many of the first state entries should be ignored (0 means all are used)
        endIndex - How many of the last state entries should be ignored (0 means all are used)
        """

        # Add all of the state histories within the given indices together
        beginIndex = max(beginIndex, 0)
        endIndex = max(len(objectState[3])-endIndex, 0)
        if (endIndex <= beginIndex):
            return np.zeros((2, 1))
        avgState = np.zeros((4, 1))
        for i in range(beginIndex, endIndex):
            avgState += objectState[3][i]

        # Compute and return the average velocity
        avgVelocity = (avgState/(endIndex - beginIndex))[2:]
        return avgVelocity



class Kalman:
    """
    Class for tracking an object using a Kalman filter
    """

    def __init__(self, measurement, time, qDist=1, qVel=1, r=8):
        """
        Initialize the Kalman filter, which assumes constant velocity

        measurement - The measurement data used to initialize this instance of the Kalman filter
                      (Location, Covariance) as numpy arrays
        time - The initial time at which the object was detected
        qDist - The location error introduced in the model, used to create the first two rows of 
                the Q matrix (assumed to be the same for x and y)
        qVel - The velocity error introduced in the model, used to create the last two rows of the
               Q matrix (assumed to be the same for x and y)
        r - The measurement error used to create the R matrix (assumed to be the same for x and y)
        """

        z, cov = measurement[:2]

        # Initialize arrays used for the Kalman filter
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.array([[qDist, 0, 0, 0],
                           [0, qDist, 0, 0],
                           [0, 0, qVel, 0],
                           [0, 0, 0, qVel]])
        self.R = np.identity(2) * r

        # Compute the vector for the object state and its covariance
        self.X = np.zeros((4, 1))
        self.P = np.eye(4) * (qVel * 10)
        if (z.shape[0] == 2):
            self.X[0:2, :] = z
        elif (z.shape[0] == 4):
            self.X = deepcopy(z)
        if ((cov.shape[0] == 2) and (cov.shape[1] == 2)):
            self.P[0:2, 0:2] = cov
        elif ((cov.shape[0] == 4) and (cov.shape[1] == 4)):
            self.P = deepcopy(cov)

        # Store time and flag for checking if an extrapolation has happened
        self.TIME = time
        self.UPDATE = time
        self.EFLAG = False

        # Set up deque to store state history
        self.STATEHIST = deque(maxlen=20)

        # Set up variables for computing the score of a state
        self.MOTIONSCORE = self.getMotionScore(measurement)
        self.APPEARANCESCORE = 0
        self.MOTION_W = 1
        self.APPEARANCE_W = 0


    def extrapolate(self, time):
        """
        Project the object forward in time

        time - The time to which it should be projected (i.e. if the previous time was 1 and the 
               object should be projected forward by one second, the value of 'time' should be 2)
        """

        # Move the point forward by the given time step and update the covariance
        F = np.array([[1, 0, time - self.TIME, 0],
                      [0, 1, 0, time - self.TIME],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        x_ = np.matmul(F, self.X)
        P_ = np.matmul(np.matmul(F, self.P), F.transpose()) + self.Q

        # Save all of the variables
        self.X = x_
        self.P = P_
        self.TIME = time
        self.EFLAG = True
        self.STATEHIST.append(x_)

        # Update the motion score as if no detection was found (changed if a detection is found)
        self.MOTIONSCORE = self.getMotionScore([self.X])


    def update(self, measurement, time):
        """
        Update the object's projected location with a measurement

        measurement - The measured location and covariance (x, P) of the object
        time - The time at which the measurement was taken
        """

        # Check to see if the object has been extrapolated, otherwise perform the operation
        if (not self.EFLAG):
            self.extrapolate(time)

        # Calculate innovation and innovation covariance
        inno = self.innovation(measurement)
        S = self.innovationCovar()

        # Update position and covariance
        K = np.matmul(np.matmul(self.P, self.H.transpose()), np.linalg.inv(S))*1
        x = self.X + np.matmul(K, inno)
        P = np.matmul((np.eye(4) - np.matmul(K, self.H)), self.P)

        # Update the motion and appearance scores
        self.MOTIONSCORE = self.getMotionScore(measurement)

        # Store all of the variables
        self.X = x
        self.P = P
        self.TIME = time
        self.UPDATE = time
        self.EFLAG = False


    def setVariables(self, measurement, time):
        """
	Sets the location, covariance and time to the specified values. This function should only
        be used to restart a track that has been suspended, not to add a new detection.

	measurement - The measured location and covariance (x, P) of the object as numpy arrays
        time - The time at which the object was detected
        """

        z, cov = measurement[:2]

        # Set the location and covariance for the object
        self.P = np.eye(4) * 1e6
        if (z.shape[0] == 2):
            self.X[0:2, :] = z
        elif (z.shape[0] == 4):
            self.X = deepcopy(z)
        if ((cov.shape[0] == 2) and (cov.shape[1] == 2)):
            self.P[0:2, 0:2] = cov
        elif ((cov.shape[0] == 4) and (cov.shape[1] == 4)):
            self.P = deepcopy(cov)

        # Set the time and extrapolation flag for the object
        self.TIME = time
        self.EFLAG = False


    def innovation(self, data):
        """
        Calculate the innovation for the object with the given measurement

        data - The measured location and covariance (x, P) of the object
        """

        z, P = data
        innovation = z - np.matmul(self.H, self.X)
        return innovation


    def innovationCovar(self):
        """
        Calculate the innovation covariance for the object
        """

        S = np.matmul(np.matmul(self.H, self.P), self.H.transpose()) + self.R
        return S


    def getLocation(self):
        """
        Returns the current object location
        """

        return self.X.flatten()[:2]


    def getVelocity(self):
        """
        Returns the velocity for the object
        """

        return self.X.flatten()[2:]


    def getCovariance(self):
        """
        Returns the current covariance for the object location
        """

        return self.P


    def getTime(self):
        """
        Returns the last time at which this object was extrapolated or updated
        """

        return self.TIME


    def getLastUpdate(self):
        """
        Returns the last time at which this object was updated
        """

        return self.UPDATE


    def getAverageVelocity(self, numSteps):
        """
        Returns the average velocity for the last number of time steps, after removing the given 
        number of most recent number of time steps.

        numSteps - The number of most recent time steps that should be ignored in the computation
        """

        if (len(self.STATEHIST) == 1):
            velList = list(self.STATEHIST)
        else:
            velList = list(self.STATEHIST)[:-min(numSteps, len(self.STATEHIST) - 1)]
        avgVel = np.zeros((2))
        for v in velList:
            avgVel += v.flatten()[2:]
        avgVel /= len(velList)
        if (len(velList) == 0):
            print(velList)
            print(self.STATEHIST)
            print(avgVel)
            print()
        return avgVel


    def distance(self, data):
        '''
        Compute the Mahalanobis distance between a given point and distribution

        data - The location of a detected object in the format (x, y)
        '''

        innov = self.innovation(data)
        innovCov = self.innovationCovar()
        dist = np.matmul(np.matmul(np.transpose(innov), np.linalg.inv(innovCov)), innov)
        return np.sqrt(dist[0, 0])


    def getScore(self):
        '''
        Compute an error for the similarity between the track and a measurement

        detection - The location of a detected object in the format (x, y)
        pTrue - The probablility that the detection is a true detection
        '''

        return self.MOTION_W * self.MOTIONSCORE + self.APPEARANCE_W * self.APPEARANCESCORE


    def getMotionScore(self, measurement):
        """
        Returns the motion score between the current Kalman state estimate and the given 
        measurement, using a normal distribution

	measurement - The measured location and covariance (x, P) of the object as numpy arrays
        """

        return np.log(normal.pdf(x=measurement[0].flatten()[:2], mean=self.X.flatten()[:2], 
                                 cov=self.P[0:2, 0:2]))


    def __repr__(self):
        """
        String representation for Kalman filter.
        """

        return "Kalman - (State: [%.2f, %.2f] Time: %d Weight: %.5f)" % (self.X[0][0], self.X[1][0], self.TIME, -1.0/self.getScore())


class MultiKalman:
    """
    Class for tracking multiple objects using a Kalman filter
    """

    def __init__(self, updateCount):
        """
        Initialize the tracker

        updateCount - The number of iterations an object can be extrapolated without being updated.
        """

        self.trackList = {}
        self.suspendList = {}
        self.updateHist = {}
        self.extrapHist = {}
        self.updateCount = updateCount


    def newObject(self, ID, z, cov, time, qDist=1, qVel=1, r=8):
        """
        Add a new object to be tracked

        ID - A unique identifier for the object
        z - The measured location of the object as a numpy array
        cov - The covariance of the detection as a numpy array
        time - The time at which the object was detected
        qDist - The location error introduced in the model, used to create the first two rows of 
                the Q matrix (assumed to be the same for x and y)
        qVel - The velocity error introduced in the model, used to create the last two rows of the
               Q matrix (assumed to be the same for x and y)
        r - The measurement error used to create the R matrix (assumed to be the same for x and y)
        """

        # Add an entry/"object" for the given ID
        self.trackList[ID] = [Kalman(z, cov, time, qDist, qVel, r), 0]
        self.updateHist[ID] = [tuple(self.trackList[ID][0].getLocation()[:2])]
        self.extrapHist[ID] = [tuple(self.trackList[ID][0].getLocation()[:2])]


    def extrapolate(self, time):
        """
        Project all objects forward in time, and remove objects/clutter that haven't been updated
        for the previously specified number of iterations.

        time - The time to which it should be projected (i.e. if the previous time was 1 and the 
               object should be projected forward by one second, the value of 'time' should be 2)
        """

        # Iterate through all objects that are currently being tracked
        for trackID in self.trackList.keys():

            K, c = self.trackList[trackID]

            # Remove objects that have gone too long without an update
            if c >= self.updateCount:
                self.trackList.pop(trackID)
                continue

            # Extrapolate the object by the given time step
            K.extrapolate(time)

            self.trackList[trackID] = [K, c+1]
            self.extrapHist[trackID].append(tuple(K.getLocation()[:2]))

        # Iterate through all suspended objects and increase the time they've been suspended
        for trackID in self.suspendList.keys():

            track, count = self.suspendList[trackID]
            if count == 0:
                self.suspendList.pop[trackID]
            else:
                self.suspendList[trackID] = [track, count - 1]


    def update(self, trackID, measurement, time):
        """
        Update the object's projected location with a measurement

        trackID - The ID of the object
        measurement - The measured location of the object
        """

        # Update the track
        K, c = self.trackList[trackID]
        K.update(measurement, time)

        # Update storage variables
        self.trackList[trackID] = [K, 0]
        self.updateHist[trackID].append(tuple(K.getLocation()[:2]))

        return K.getLocation(), K.getCovariance()


    def hasID(self, trackID):
        """
        Check to see if the specifiec object is currently being tracked

        trackID - The ID of the object
        """

        return trackID in self.trackList


    def getLastPos(self, trackID):
        """
        Return the last known position of the object

        trackID - The ID of the object
        """

        x, P, t, c = self.trackList[trackID]
        return x, P


    def suspendTrack(self, trackID, count):
        """
        Temporarily suspends the track for the given object, so that it is no longer updated, but 
        also not removed for the given number of iterations.

        trackID - The ID of the object that should be suspended
        count - The number of iterations the item should be suspended for before being removed
        """

        if trackID in self.trackList:
            track = self.trackList.pop(trackID)[0]
            self.suspendList[trackID] = [track, count]


    def restartTrack(self, trackID, measurement, time):
        """
        Restarts tracking an object if it is suspended.

        trackID - The ID of the object for which tracking should be restarted
        measurement - The measurement with which the suspended track should be associated
        """

        if trackID in self.suspendList:
            track, count = self.suspendList.pop(trackID)
            z, P = measurement
            track.setVariables(z, track.getCovariance(), time)
            self.trackList[trackID] = [track, 0]

    def getUpdateHist(self, trackID):
        """
        Return the history of all updated locations of a given track 

        trackID - The ID of the track for which the history is sought
        """

        if (trackID in self.updateHist):
            return self.updateHist[trackID]
        return []


    def getExtrapolatedHist(self, trackID):
        """
        Return the history of all extrapolated locations of a given track 

        trackID - The ID of the track for which the history is sought
        """

        if (trackID in self.extrapHist):
            return self.extrapHist[trackID]
        return []
