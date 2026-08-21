import cv2
import subprocess

# import OpenCV Zoo's palm and hand pose detection classes
from mp_palmdet import MPPalmDet
from mp_handpose import MPHandPose

def playPauseMusic():
    subprocess.run([
        "osascript",
        "-e",
        'tell application "Music" to playpause'
    ])

def nextSong():
    subprocess.run([
        "osascript",
        "-e",
        'tell application "Music" to next track'
    ])

def previousSong():
    subprocess.run([
        "osascript",
        "-e",
        'tell application "Music" to previous track'
    ])

# open the Mac's camera
videoCapture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# load the palm detection model
palmDetector = MPPalmDet(
    modelPath="palm_detection_mediapipe_2023feb_int8bq.onnx",
    scoreThreshold=0.6,
    nmsThreshold=0.3
)

# load the hand landmark detection model
handPoseDetector = MPHandPose(
    modelPath="handpose_estimation_mediapipe_2023feb_int8bq.onnx"
)

# Load the face detection model
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2026may.onnx",
    "",
    (320, 320),
    0.8
)

# Stores the previous gesture so the same gesture
# does not repeatedly activate an Apple Music command
lastGesture = None

# Continuously read frames from the camera
while True:
    ret, frame = videoCapture.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # Face detection
    detector.setInputSize((width, height))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

    # palm detection
    palms = palmDetector.infer(frame)

    totalFingerCount = 0

    for palm in palms:
        hand = handPoseDetector.infer(frame, palm)

        if hand is not None:
            landmarks = hand[4:67].reshape(21, 3)

            fingerCount = 0

            # 8  = index
            # 12 = middle
            # 16 = ring
            # 20 = pinky
            # ignore thumb

            fingerTips = [8, 12, 16, 20]
            
            # landmark at the base of each finger
            fingerBases = [5, 9, 13, 17]

            for tip, base in zip(fingerTips, fingerBases):
                if landmarks[tip][1] < landmarks[base][1]:
                    fingerCount += 1

            totalFingerCount += fingerCount

            # draw hand landmarks
            for point in landmarks:
                x = int(point[0])
                y = int(point[1])

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (255, 255, 255),
                    -1
                )

    # Apple Music controls
    if len(palms) > 0:

        if totalFingerCount != lastGesture:

            if totalFingerCount == 1:
                playPauseMusic()

            elif totalFingerCount == 2:
                nextSong()

            elif totalFingerCount == 3:
                previousSong()

            lastGesture = totalFingerCount

    else:
        lastGesture = None

    # display finger count
    cv2.putText(
        frame,
        f"Fingers: {totalFingerCount}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

videoCapture.release()
cv2.destroyAllWindows()