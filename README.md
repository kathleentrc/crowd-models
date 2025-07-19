# Crowd Models

Run the pipeline using:
```
python main.py
```

The main file serves as the entry point for the crowd density estimation application. It is used to test the entire methodology pipeline, which includes:
- Image processing using YOLOv8
- OpenCV Gaussian Blur
- Multimodal Fusion with Early Fusion
- Bayesian Neural Network Initial Training
- Spatio-Temporal Graph Convolutional Network (STGCN)
  
***Note: The Spatio-Temporal Graph Convolutional Network (STGCN) is not yet implemented in this version.***

Each process is stored in the ```src/``` directory.
