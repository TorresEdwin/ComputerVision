# Computer Vision Gesture Controls

A real-time computer vision application that uses hand gesture recognition to control media playback through a webcam. The application detects and tracks hand gestures and maps them to system actions such as playing, pausing, and controlling media.

## Features

- Real-time hand detection and tracking using a webcam
- Hand gesture recognition with MediaPipe
- Gesture-based media controls
- Play and pause music without using a keyboard or mouse
- Real-time OpenCV video visualization

## Technologies

- Python
- OpenCV
- MediaPipe
- NumPy

## How It Works

The application captures live video from the user's webcam using OpenCV. Each frame is processed by a hand detection and pose estimation pipeline to identify the position of the hand and its landmarks.
Recognized hand gestures are then mapped to media control commands, allowing the user to interact with their computer without a keyboard or mouse.
