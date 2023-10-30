import numpy as np
import sys
import os

from Model import BaseKalman as Kalman
from MHT import MHT

if __name__ == '__main__':

    # Read which toy example should be used
    if (len(sys.argv) == 1):
        example = 'toy1.txt'
    else:
        example = sys.argv[1]
    if (not os.path.isfile(example)):
        print("Example '" + example + "' not found...")
        exit()

    # Read the data from the toy example (Only object points)
    t = -1
    detections = {}
    baseCov = np.identity(4)
    baseCov[2, 2] = 200
    baseCov[3, 3] = 200
    with open(example, 'r') as toy:
        for line in toy:
            if (len(line) == 0) or (line[0] == '#'):
                continue
            data = line.split()
            if (len(data) == 1):
                t = int(data[0])
                detections[t] = []
            elif (len(data) == 2):
                x = float(data[0])
                y = float(data[1])
                detections[t].append((np.array([[x], [y]]), baseCov))

    # Initialize an instance of the MHT and iterate through the data
    tracker = MHT(model=Kalman, 
                  gateDistance=5, 
                  timeDrop=5, 
                  minTrackUpdates=5, 
                  pNewTrack=0.004, 
                  pFalse=0.00002,
                  pDetect=0.999, 
                  endLambda=20,
                  nScan=3, 
                  maxHypotheses=100,
                  minHypothesisRatio=0.001,
                  timeInterval=1,
                  debug=True)
    times = list(detections.keys())
    times.sort()
    for t in times:
        tracker.iterate(detections[t], t)

    print("\n\nCurrent Tracks")
    current = tracker.getCurrentTracks()
    for c in current:
        current[c].getRoot().printTree()

    print("\n\nEnded Tracks")
    ended = tracker.getEndedTracks()
    for e in ended:
        ended[e].getRoot().printTree()
