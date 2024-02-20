# Face Detection using OpenCV-Python

### Technologies Used:

- OpenCV-Python
- Numpy
- Scikit Learn

### Required Extras:

- Haarcascade Classifiers
- YuNet Model

### Upcoming

- Interactive Menu
- More Models

#### Step 01:

Clone and Enter the Repository

```shell script
git clone https://github.com/TheProjectsX/opencv-face-detection
cd opencv-face-detection
```

#### Step 02:

Create a Python Virtual Environment and Activate it

```shell script
python -m venv py-venv

# For Windows:
.\py-venv\Scripts\activate

# For Linux:
./py-venv/Scripts/activate
```

#### Step 03:

Install necessary Packages

```shell script
pip install opencv-python scikit-learn
```

### Step 04:

Select a Model

- Haarcascade Classifier is a Classic light-weight model. Can Detect faces but not as good as YuNet.
- YuNet Model. Can detect more Faces than Haarcascade.

For Haarcascade:

```shell script
cd Haarcascade
```

For YuNet

```shell script
cd YuNet
```

#### Step 05:

Train the Model.

```shell script
python train_face_data_vid.py --source <Frame-Source> --source-path <Frame-Source-Path>
```

- `<Frame-Source>` has 2 options: `image` and `video`
- `<Frame-Source-Path>` can be:
  - Either an Image File Path (for --source image)
  - Folder Path (for both sources)
  - A Video File Path (for --source video)
  - Camera Index (for --source video)

**Additional Options**

- `--upgrade` use to Upgrade / Re-Train any Existing person's model
- `--update` use to Update an Existing person's model

If Model does not exists, will create a New Model
