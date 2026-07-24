# denoise_kalman
This code is directly related to the manuscript currently submitted by us to PeerJ Computer Science. Readers are welcome to cite this relevant manuscript: "Enhanced Video Denoising and Detail Restoration via Spatially Adaptive Kalman Filtering".
README
======

Title
-----
Enhanced Video Denoising and Detail Restoration via Spatially Adaptive Kalman Filtering

Code Repository
---------------
https://github.com/395822132-stack/denoise_kalman.git

Code File
---------
kalman_filter_video_stabilization.py

Overview
--------
This repository contains the Python code used for video stabilization in the
study. The program detects and tracks feature points between consecutive video
frames, estimates global frame-to-frame translation, smooths the estimated
motion using a Kalman filter, and applies an affine transformation to generate
stabilized output videos. It also records processing time and processing speed
for each video.

Dataset
-------
The experiments use the third-party Rocket dataset hosted by Science Data Bank
(ScienceDB).

Dataset URL:
https://www.scidb.cn/anonymous/QU5qeWEy

The dataset was accessed on 24 July 2026. The dataset is not redistributed in
this repository. Users should download it from the URL above and organize the
video files using the directory structure described below.

Software Requirements
---------------------
Python 3.9 or later

Required Python packages:
- opencv-python
- numpy
- tqdm

Install the required packages with:

pip install opencv-python numpy tqdm

Input Data Structure
--------------------
The program expects the training videos to be organized into five class
directories:

rocket_dataset/
└── train/
    ├── 001/
    ├── 002/
    ├── 003/
    ├── 004/
    └── 005/

Supported video formats are MP4, AVI, MOV, and MKV.

Before running the program, open kalman_filter_video_stabilization.py and set
the input and output directories in the main() function. The input path must
refer to the training-set root directory rather than to an individual video
file:

train_input_dir = "./rocket_dataset/train"
train_output_dir = "./rocket_dataset_stabilized/train"

Running the Code
----------------
Run the program from the repository directory:

python kalman_filter_video_stabilization.py

The program processes all supported videos located in the 001, 002, 003, 004,
and 005 subdirectories.

Method
------
For each video, the first frame is converted to grayscale and feature points
are detected using ORB. When fewer than 50 ORB feature points are detected,
Shi-Tomasi feature detection is used as a fallback with a maximum of 200
corners, a quality level of 0.01, a minimum distance of 7 pixels, and a block
size of 7 pixels.

Feature points are tracked between consecutive frames using pyramidal
Lucas-Kanade optical flow with a window size of 15 × 15 pixels, two pyramid
levels, a maximum of 10 iterations, and a termination threshold of 0.03. At
least four valid tracked points are required. Horizontal and vertical global
motion are estimated from the median displacement of the valid feature points.

The estimated horizontal and vertical translations are smoothed using a Kalman
filter with a four-dimensional state and a two-dimensional measurement. The
process-noise covariance is set to 1 × 10^-4, and the measurement-noise
covariance is set to 1 × 10^-2. The filtered translation is applied to each
frame using a two-dimensional affine transformation.

When feature detection or optical-flow tracking fails, the original frame is
written to the output video and feature points are detected again for the next
frame.

Output
------
The processed videos are saved in the following structure:

rocket_dataset_stabilized/
└── train/
    ├── 001/
    ├── 002/
    ├── 003/
    ├── 004/
    ├── 005/
    └── performance_stats.txt

The output videos retain the original frame width, frame height, and frame
rate and are encoded using the MP4V codec. The performance_stats.txt file
reports the processed frame count, total processing time, processing speed in
frames per second, and real-time processing ratio for each video.

Reproducibility
---------------
To reproduce the processing procedure:

1. Download the Rocket dataset from the ScienceDB URL provided above.
2. Arrange the videos in the documented class-directory structure.
3. Install Python and the three required packages.
4. Set train_input_dir and train_output_dir in the main() function.
5. Run kalman_filter_video_stabilization.py using the command shown above.

The algorithm uses deterministic parameter settings for feature detection,
optical-flow tracking, motion estimation, and Kalman filtering. No random
training procedure is used in this script. Runtime may vary with the computer
hardware, operating system, OpenCV build, video resolution, and codec support.

Code Availability
-----------------
The computer code used in this study is publicly available at:

https://github.com/doctordidi/denoise_kalman

The file kalman_filter_video_stabilization.py is provided for review and
publication with the manuscript.

