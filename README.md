# Iris Classification – k-Nearest Neighbors (k-NN)

A simple machine learning classification project using the Iris dataset and k-Nearest Neighbors (k-NN). Includes data exploration, visualization, model training, and evaluation using scikit-learn.

This project demonstrates a basic machine learning workflow using the famous **Iris dataset**. It includes:
- Loading dataset from **scikit-learn**
- Exploring the data structure (features, labels)
- Splitting data into train and test sets
- Data visualization with **pandas** and **matplotlib**
- Training a **k-Nearest Neighbors (k-NN)** classifier
- Making predictions and evaluating model accuracy

Dataset
- **150** flower samples
- **3** species:
  - Setosa
  - Versicolor
  - Virginica
- **4 features**:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)


Technologies used
- scikit-learn
- numpy 
- pandas 
- matplotlib 
- Python

Model
- Algorithm: **k-Nearest Neighbors**
- Parameter: `n_neighbors = 1`
- Accuracy on test set: **~97%**

How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/iris-knn-classification.git

cd iris-knn-classification

# Install dependencies
pip install -r requirements.txt

# Run the script
python iris_knn.py
