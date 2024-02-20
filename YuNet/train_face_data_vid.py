import cv2
import numpy as np
import time
import json
import os
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
CurrentTime = str(int(time.time()))


# Array to save Face Data
face_data_list = []

# Window Name
windowName = "Train Face"

# Frame Size
NpArraySize = [100, 100, 3]

# Detection Count Limit
DetectionCountLimit_vid = 10
DetectionCountLimit_img = 4

# Minimum Images
MinImageCount = 1


# We will parse the Images Count using this Function.  images are less than N, will repeat from the start to make N
def parseListToN(targetList, n=DetectionCountLimit_img):
    if (len(targetList) == n):
        return targetList

    repetitions = (n // len(targetList)) + 1

    extendedList = targetList * repetitions
    return extendedList


# Upgrade Model Info
def upgradeModel(modelName):
    try:
        detectionModel = json.load(
            open(FilePath + "modelDetection.json", "r"))
    except:
        return

    detectionKey = next(
        (key for key, val in detectionModel.items() if val == modelName), None)
    if (detectionKey is None):
        print("Model Does not Exist. Creating new Model", flush=1)
        return None

    globals()["CurrentTime"] = detectionKey

    return detectionKey

# Upgrade Model Info


def updateModel(modelName):
    detectionKey = upgradeModel(modelName)
    if (detectionKey is None):
        return

    if not (os.path.isfile(FilePath + detectionKey + ".npy")):
        print("Model Dataset Does not Exists")
        return

    loaded_array = np.load(FilePath + detectionKey + ".npy")
    copiedArr = NpArraySize.copy()
    copiedArr.insert(0, loaded_array.shape[0])

    originalSize = tuple(copiedArr)
    globals()["face_data_list"] = loaded_array.reshape(
        originalSize).tolist()
    # 100, 100, 3 is the resized cv2 frame


# Detect Face from an Image / Frame
def detectFace(image):
    faces = []
    results = FaceModel.infer(image)

    for result in results:
        bbox = result[0:4].astype(np.int32)
        faces.append(bbox)

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    return faces


# Save the Data
def storeTrainData(modelName):
    # Create folder if not Exist
    if not (os.path.isdir(FilePath)):
        os.mkdir(FilePath)

    # Store the Data to File
    npArray = np.array(face_data_list)
    npArray = npArray.reshape(npArray.shape[0], -1)

    np.save(FilePath + CurrentTime, npArray)

    modelDetection = {}

    if (os.path.isfile(FilePath + "modelDetection.json")):
        modelDetection = json.load(open(FilePath + "modelDetection.json", "r"))

    modelDetection[CurrentTime] = modelName
    json.dump(modelDetection, open(FilePath + "modelDetection.json", "w"))

    print("\nModel: " + modelName)
    print("Data Saved In: " + FilePath + CurrentTime + ".npy")


# Iteration Count
i = 0


# Store Face Data and Return Frame with Face Pointed
def storeAndPointFace(frame, faces, count=None):
    globals()["i"] += 1

    # Only use the Largest Detected Face
    face = faces[0]
    x, y, w, h = face
    faceArea = frame[y: y+h, x:x+w]
    faceResized = cv2.resize(faceArea, tuple(NpArraySize[:2]))

    # Will save DetectionCountLimit Frames of the Face, Each after 10 leaps
    if (args.source == "video"):
        if (i <= 10 * DetectionCountLimit_vid) and (i % 10 == 0):
            face_data_list.append(faceResized)
    else:
        face_data_list.append(faceResized)

    if (count == None):
        count = str(len(face_data_list))

    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 3)
    frame = cv2.putText(frame, "Stored: " + count,
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    return frame


# Video Model Capture
def captureVidModel(modelName, videoSource=0):
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
            frame = storeAndPointFace(frame, faces)

        cv2.imshow(windowName, frame)

        # Break if Iteration count is 10 * DetectionCountLimit
        if (i == 10 * DetectionCountLimit_vid):
            break

        # Break the loop if ESC pressed or Window Closed manually
        if (cv2.waitKey(1) & 0xFF == 27):
            print("\n[!] Operation Ended by User...")
            exit("\nExiting program...")

        if (cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1):
            print("\n[!] Operation Ended by User...")
            exit("\nExiting program...")

    # Store Train Data
    storeTrainData(modelName)

    # Release Video Capture Frame
    cap_vid.release()


# Image Model Capture
def captureImgModel(modelName, filePath):
    imgExtn = [".jpg", "jpeg", ".png"]

    if (os.path.isdir(filePath)):
        fileList = [filePath + "/" + x for x in os.listdir(filePath)]
    else:
        fileList = [filePath]

    fileList = [x for x in fileList if (x[-4:] in imgExtn)]

    if (len(fileList) < MinImageCount):
        print("\n[!] At least " + str(MinImageCount) + " Images are Required")
        print("[!] Supported Image formats are: " +
              ", ".join(imgExtn).replace(".", ""))
        exit("\nExiting program...")

    # Get DetectionCountLimit Images only
    fileList = parseListToN(fileList)

    imgParsed = []
    # Iterate over the images and perform operation
    for i in range(len(fileList)):
        filePath = fileList[i]
        img = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
        height, width = img.shape[:2]
        FaceModel.setInputSize([width, height])

        faces = detectFace(img)

        if (len(faces) == 0):
            continue

        imgParsed.append(fileList[i])
        img = storeAndPointFace(img, faces, count=str(len(imgParsed)))

        cv2.imshow(windowName + " " + str(len(imgParsed)), img)

        if (len(imgParsed) == DetectionCountLimit_img):
            break

    if (len(imgParsed) == 0):
        print("\n[!] No Face Detected in given Images...")
        exit("\nExiting program...")

    parseLeft = parseListToN(
        imgParsed, n=DetectionCountLimit_img - len(imgParsed))
    if (len(imgParsed) == DetectionCountLimit_img):
        parseLeft.clear()

    # Iterate over the images and perform operation
    for i in range(len(parseLeft)):
        filePath = parseLeft[i]
        img = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detectFace(gray_img)

        imgParsed.append(fileList[i])
        img = storeAndPointFace(img, faces, count=str(len(imgParsed)))

        cv2.imshow(windowName + " " + str(len(imgParsed)), img)

    # Store Train Data
    storeTrainData(modelName)

    # Wait for Key Press
    cv2.waitKey(0) & 0xFF


# Creating an Argument Parser
parser = argparse.ArgumentParser(
    description="Create Dataset from an Image or Video for Detection")

parser.add_argument(
    '--source', help="Specify if Image source or Video source - `image` or `video`", default="video")
parser.add_argument(
    '--source-path', help="Specify the Source Path - File / Folder path or Camera Index", default="0")
parser.add_argument(
    '--upgrade', action="store_true", help="Upgrade Existing Model or Create New", required=False)
parser.add_argument(
    '--update', action="store_true", help="Update Existing Model or Create New", required=False)

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

# Get the Name of the Model
currentModelName = input("\nEnter Model Name:> ")
if (currentModelName == ""):
    print("\nModel Name cannot be Empty")
    exit("\nExiting program...")

# If we want to upgrade or update Our existing Model, we will just simply change the time
if (args.upgrade):
    print("\n[Action]: Upgrading Existing Model")
    upgradeModel(currentModelName)
elif (args.update):
    print("\n[Action]: Updating Existing Model")
    updateModel(currentModelName)


# Start Capturing Process
print("\nStarting Capture Process...")
if (args.source == "video"):
    captureVidModel(currentModelName, sourcePath)
elif (args.source == "image"):
    captureImgModel(currentModelName, sourcePath)

# Release Video Frame and Destroy Windows
cv2.destroyAllWindows()

print("\nDetection Ended: " + str(len(face_data_list)))
