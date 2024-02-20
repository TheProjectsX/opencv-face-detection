import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import json
import argparse
from scripts.yunet import YuNet

# Cascade Model
# face_detection_yunet_2023mar_int8.onnx is for far face detection
FaceModel = YuNet(modelPath="Models/face_detection_yunet_2023mar.onnx",
                  inputSize=[320, 320],
                  confThreshold=0.9,
                  nmsThreshold=0.3,
                  topK=5000,
                  backendId=3,
                  targetId=0)

# File Save Path
FilePath = f"./Dataset/"

# Array to save Face Data
face_data_list = np.array([])
face_label_list = np.array([])
face_detection = {}

# Window Name
windowName = "Live Face Detection"

# KNN Algorithm Class
KNN_classifier = KNeighborsClassifier(n_neighbors=4)

# Resize Image according to Aspect Ratio from given Height or Width


def resizeImage(image, width=None, height=None):
    # Get the original dimensions
    original_height, original_width, channels = image.shape

    # Calculate the missing dimension based on the aspect ratio
    if width is not None:
        new_width = width
        new_height = int((original_height / original_width) * new_width)
    elif height is not None:
        new_height = height
        new_width = int((original_width / original_height) * new_height)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


# Detect Face from an Image / Frame
def detectFace(image):
    faces = []
    results = FaceModel.infer(image)

    for result in results:
        bbox = result[0:4].astype(np.int32)
        faces.append(bbox)

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    return faces


# Load Train Model Data and Save in the Array
def loadAndStoreModels():
    if not (os.path.isdir(FilePath)):
        return False

    fileList = os.listdir(FilePath)
    if (len(fileList) == 0):
        return False

    for fx in os.listdir(FilePath):
        if not (fx.endswith('.npy')):
            continue

        data_item = np.load(FilePath + fx)
        if (len(face_data_list) == 0):
            globals()["face_data_list"] = data_item
        else:
            globals()["face_data_list"] = np.concatenate(
                (face_data_list, data_item), axis=0)

        name = fx[:-4]
        names = np.array([name] * len(data_item))
        globals()["face_label_list"] = np.concatenate(
            (face_label_list, names), axis=0)

    try:
        globals()["face_detection"] = json.load(
            open(FilePath + "modelDetection.json", "r"))
    except:
        print("\n[!] Model Detection File Is not Correct")
        exit("\nExiting program...")

    return True


# Recognize Face and Return Frame
def recognizeFace(frame, faces):
    for face in faces:
        x, y, w, h = face
        faceArea = frame[y: y+h, x:x+w]
        faceResized = cv2.resize(faceArea, (100, 100)).flatten().reshape(1, -1)

        detector = KNN_classifier.predict(faceResized)[0]

        detectedName = face_detection[detector]

        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 3)
        frame = cv2.putText(frame, detectedName,
                            (x+2, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    return frame


# Detect From Video
def detectFromVid(videoSource=0):
    # Initialize Camera
    cap_vid = cv2.VideoCapture(videoSource)

    if not (cap_vid.isOpened()):
        if (type(videoSource) == int):
            print("\n[!] Wrong Camera Index Given")
        else:
            print("\n[!] Wrong File Path Given")

        exit("\nExiting program...")

    width = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    FaceModel.setInputSize([width, height])

    # Start the Loop
    while cap_vid.isOpened():
        ret, frame = cap_vid.read()
        if (not ret):
            continue

        faces = detectFace(frame)

        if not (len(faces) == 0):
            frame = recognizeFace(frame, faces)

        cv2.imshow(windowName, frame)

        # Break the loop if ESC pressed or Window Closed manually
        if (cv2.waitKey(1) & 0xFF == 27):
            break

        if (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1):
            break

    # Release Video Capture Frame
    cap_vid.release()


# Detect from Image
def detectFromImg(filePath):
    imgExtn = [".jpg", "jpeg", ".png"]
    if (os.path.isdir(filePath)):
        fileList = [filePath + "/" + x for x in os.listdir(filePath)]
    else:
        fileList = [filePath]

    fileList = [x for x in fileList if (x[-4:] in imgExtn)]

    if (len(fileList) == 0):
        print("\n[!] Wrong Image Format")
        print("[!] Supported Image formats are: " +
              ", ".join(imgExtn).replace(".", ""))
        exit("\nExiting program...")

    for i in range(len(fileList)):
        img = cv2.imread(fileList[i], cv2.IMREAD_UNCHANGED)

        height, width = img.shape[:2]
        FaceModel.setInputSize([width, height])

        faces = detectFace(img)

        if not (len(faces) == 0):
            img = recognizeFace(img, faces)
        else:
            print("\nNo Face Detected in Image " + str(i + 1))

        w, h, c = img.shape
        if (h > 600):
            img = resizeImage(img, height=600)

        cv2.imshow(windowName + " - " + str(i + 1), img)

    # Wait for Key Press
    cv2.waitKey(0) & 0xFF


# Creating an Argument Parser
parser = argparse.ArgumentParser(
    description="Detect Faces from an Image or Video for Detection")

parser.add_argument(
    '--source', help="Specify if Image source or Video source - `image` or `video`", default="video")
parser.add_argument(
    '--source-path', help="Specify the Source Path - File / Folder path or Camera Index", default="0")

args = parser.parse_args()

if (args.source == "video"):
    if (args.source_path.isnumeric()):
        sourcePath = int(args.source_path)
    elif not (os.path.isfile(args.source_path)):
        print("\n[!] Source Path must be a Video File or Camera Index")
        exit("\nExiting program...")
    else:
        sourcePath = args.source_path
elif (args.source == "image"):
    if not (os.path.exists(args.source_path)):
        print("\n[!] Source Path is Wrong")
        exit("\nExiting program...")
    else:
        sourcePath = args.source_path
else:
    print("\n[!] --source value must be `image` or `video`")


# Load and Store the Models
print("\nLoading Models...", flush=1)
loadAndStoreModels()

# Train the KNN
print("\nAdding Train Datasets...", flush=1)
KNN_classifier.fit(face_data_list, face_label_list)

# Start Detection Process
print("\nStarting Detection Process...")
if (args.source == "video"):
    detectFromVid(sourcePath)
elif (args.source == "image"):
    detectFromImg(sourcePath)


# Destroy all Windows
cv2.destroyAllWindows()

print("\nDetection Ended!")
