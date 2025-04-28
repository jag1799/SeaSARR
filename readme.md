# SeaSARR

---
This is an in depth analysis on the effectiveness of applying proposed image preprocessing techniques to object detection models trained on Synthetic Aperture Radar data.  Models are trained to detect all instances of a maritime vessel within an image, particularly in images with high Signal-to-Noise and Clutter-to-Noise ratios.

The dataset used can be found here: https://www.kaggle.com/datasets/kailaspsudheer/sarscope-unveiling-the-maritime-landscape

In the frcnn directory, you can find notebooks with 2 models:
1. CannyFrcnn - containing training,test and validation steps for the Canny model
2. Baseline - containing training, test and validation steps for the baseline model.

To run the project, you will need to use Python installtion is 3.8.10 or higher.
When running the notebooks for the first time, you will have to uncomment the first cell and install all required dependancies.

Once the dependancies are installed, choose the model you want to run (either the CannyFrcnn model of the baseline model) abd run all cells.

Each notebook/model contains 2 versions. One with Adam optimizer and one with SGD.

Since different machines may behave diffrently, if you encounter an index error during training, you will need to go to the utils directory, open ModelWorker.py and under the training function, inside the except block (line 112) you will need to remove this line sys.exit(-1) and replace it with 'continue' and re-run the notebook


