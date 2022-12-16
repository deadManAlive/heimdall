import json
import time

import cv2
import dlib
import imutils
import numpy as np
from imutils.video import VideoStream

from modules import thread
from modules.centroidtracker import CentroidTracker
from modules.trackableobject import TrackableObject

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

class Averager:
    def __init__(self):
        self.total = 0
        self.num = 0
    def add(self, n):
        self.total += n
        self.num += 1
    def get(self):
        return self.total / self.num

def run(mptotal_down, mptotal_up):
    cfg = json.load(open("config.json"))

    net = cv2.dnn.readNetFromCaffe(cfg["s_prototxt"], cfg["s_model"])

    if not cfg["b_debug"]:
        print("[INFO] Starting the live stream..")
        vs = VideoStream(cfg["i_camaddr"]).start()
        time.sleep(2.0)

    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(cfg["s_dbgvideo"])

    # width = None
    # height = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackable_objects = {}

    cdef unsigned long total_frames = 0
    cdef unsigned long frame_skip = cfg["i_fskip"]
    cdef float conf = cfg["f_confidence"]
    cdef int total_down = 0
    cdef int total_up = 0
    cdef int inside = 0
    cdef int width = 0
    cdef int height = 0

    if cfg["b_thread"]:
        vs = thread.ThreadingClass(cfg["i_camaddr"])

    tavg = Averager()

    bDBG = cfg["b_debug"]
    sDBV = cfg["s_dbgvideo"]

    while True:
        frame = vs.read()
        frame = frame[1] if bDBG else frame

        if sDBV is not None and frame is None:
            break
        
        ###
        tstart = time.perf_counter_ns()
        ###

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if width == 0 or height == 0:
            (height, width) = frame.shape[:2]

        rects = []

        if total_frames % frame_skip == 0:
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (width, height), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > conf:
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    trackers.append(tracker)

        else:
            for tracker in trackers:
                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackable_objects.get(objectID, None)

            if to is None:
                to = TrackableObject(objectID, centroid)

            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < height // 2:
                        total_up += 1
                        to.counted = True

                    elif direction > 0 and centroid[1] > height // 2:
                        total_down += 1
                        to.counted = True

                    inside = total_down - total_up
                    # TODO: optimization: inside is unnecessary sync'd object.

            trackable_objects[objectID] = to

        # TODO: optimize implementation (move outside main loop?)

        elap = time.perf_counter_ns() - tstart

        print(f"{total_frames}: {inside} in {elap/1000000} ms")
        mptotal_down.value = total_down
        mptotal_up.value = total_up
        total_frames += 1


    if cfg["b_thread"]:
        vs.release()
