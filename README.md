# Breast Cancer Prediction Using Feedforward Neural Network
## 1. Summary
The aim of this project is to predict cancer outcomes (with malignant or benign tumor) using feedforward neural network. The dataset used for this project can be refer https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data 
## 2. IDE and Framework
This project uses Spyder as the main IDE. The main framework used in this project are Pandas, scikit-learn and TensorFlow Keras.
## 3. Methodology
### 3.1 Data Pipeline
The data are loaded into the IDE and the unwanted column are removed. The labels are encoded such that it only contain binary number of 0 or 1 depending on the class. The data are split into 80:20 train and test ratio.
### 3.2 Model Pipeline
A feedforward neural network is constructed that are specifically catered for binary classification problem. The architecture of the model is shown in figure below.

![model](https://user-images.githubusercontent.com/92588131/164644541-fe932aeb-fb07-440a-9a9a-31018338c3a0.png)


The model is trained with a batch size of 32 and epochs 20. After training, the accuracy of the model is 99% with validation accuracy of 98%. 
## 4. Results
The training accuracy and loss, and the validation accuracy and loss are plotted in graph to better visualize the overall outcomes. The graph can is shown in these figures below.

![accuracy](https://user-images.githubusercontent.com/92588131/164644449-dc8da65a-b244-4c86-a3b7-ae929c511c45.png)
![loss](https://user-images.githubusercontent.com/92588131/164644462-31fd0d2f-236c-4fbf-8b6d-24416fa4c6bc.png)
