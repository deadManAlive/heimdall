# Heimdall
People Counting in Real-Time using live video stream/IP camera in OpenCV.

> Forked from https://github.com/saimj7/People-Counting-in-Real-Time


<div align="center">
<img src=https://imgur.com/SaF1kk3.gif" width=550>
<p>Live demo of original repo</p>
</div>

- The primary aim is to use the project as a business perspective, ready to scale.
- Use case: counting the number of people in the stores/buildings/shopping malls etc., in real-time.
- Sending an alert to the staff if the people are way over the limit.
- Automating features and optimising the real-time stream for better performance (with threading).
- Acts as a measure towards footfall analysis and in a way to tackle COVID-19.

--- 

## Table of Contents
* [Simple Theory](#simple-theory)
* [Running Inference](#running-inference)
* [Features](#features)
* [References](#references)
* [To Do](#to-do)

## Simple Theory
**SSD detector:**
- We are using a SSD (Single Shot Detector) with a MobileNet architecture. In general, it only takes a single shot to detect whatever is in an image. That is, one for generating region proposals, one for detecting the object of each proposal. 
- Compared to other 2 shot detectors like R-CNN, SSD is quite fast.
- MobileNet, as the name implies, is a DNN designed to run on resource constrained devices. For example, mobiles, ip cameras, scanners etc.
- Thus, SSD seasoned with a MobileNet should theoretically result in a faster, more efficient object detector.
---
**Centroid tracker:**
- Centroid tracker is one of the most reliable trackers out there.
- To be straightforward, the centroid tracker computes the centroid of the bounding boxes.
- That is, the bounding boxes are (x, y) co-ordinates of the objects in an image. 
- Once the co-ordinates are obtained by our SSD, the tracker computes the centroid (center) of the box. In other words, the center of an object.
- Then an unique ID is assigned to every particular object deteced, for tracking over the sequence of frames.

## Running Inference
- First up, install all the required Python dependencies:
    ```
    pip install -r requirements.txt
    ```
    > If `dlib` installation gives error, install latest Visual Studio with "Desktop development with C++" workload selected (include "C++ CMake tools for Windows") and install latest cmake library with `pip install cmake`, then install latest `dlib` with `pip install dlib`.
- To run inference on a test video file, head into the directory/use the command: 
    ```
    python main.py
    ```
- To run inference on an IP camera, first setup your camera url in `config.json`:

    ```json
    "camaddr": "http://191.138.0.100:8040/video",
    ```
    Set `"camaddr"` to `0` for webcam.
- Then run with the command:
    ```
    python main.py
    ```

## Features
***1. Threading:***
- Multi-Threading is implemented in 'modules/thread.py'. If you ever see a lag/delay in your real-time stream, consider using it.
- Threading removes OpenCV's internal buffer (which basically stores the new frames yet to be processed until your system processes the old frames) and thus reduces the lag/increases fps. 
- If your system is not capable of simultaneously processing and outputting the result, you might see a delay in the stream. This is where threading comes into action.
- It is most suitable for solid performance on complex real-time applications. To use threading, set `"thread"` to `true` in `config.json`.


***2. Scheduler:***
- Automatic scheduler to start the software. Configure to run at every second, minute, day, or Monday to Friday.
- This is extremely useful in a business scenario, for instance, you can run it only at your desired time (9-5?).
- Variables and memory would be reset == less load on your machine.

    ```python
    ## Runs at every day (9:00 am). You can change it.
    schedule.every().day.at("9:00").do(run)
    ```

***3. Timer:***
- Configure stopping the software after a certain time, e.g., 30 min or 9 hours from now.
- All you have to do is set your desired time and run the script.

    ```python
    if Timer:
        # Automatic timer to stop the live stream. Set to 8 hours (28800s).
        t1 = time.time()
        num_seconds=(t1-t0)
        if num_seconds > 28800:
            break
    ```

***4. Simple log:***
- Logs all data at end of the day.
- Useful for footfall analysis.
<img src="https://imgur.com/CV2nCjx.png" width=400>

## References
***Main:***
- Main repo: https://github.com/saimj7/People-Counting-in-Real-Time
- SSD paper: https://arxiv.org/abs/1512.02325
- MobileNet paper: https://arxiv.org/abs/1704.04861
- Centroid tracker: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

***Optional:***
- https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11
- https://pypi.org/project/schedule/

## To Do
- **Remote monitoring** (data only).
    - [x] via REST API
    - [ ] via WebSocket/Polling
- **Features**
    - [x] Remove email alert feature
    - [ ] Improve logging feature.
    - [ ] Remove (*or improve?*) trivial features: scheduler and timer.
- **Issues**
    - [ ] Multithreading not working in `debug` mode.