# importing libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Preparation of data
data = load_iris()
X = data.data
y = data.target
df = pd.DataFrame(data=X, columns=data.feature_names)
df['species'] = y

#preparing and training the model
model = Sequential([
    Dense(10, activation='relu', input_shape=(4,)),
    Dense(10, activation='relu'),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=50)

# Function to predict species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    species_index = np.argmax(prediction)
    species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    return species_map[species_index]

'''Species have 0,1,2 as values since there are 3 types of iris species in the data.
0 is for Setosa, 1 is for Versicolor, 2 is for Virginica'''
# Examples
print(predict_species(5.1, 3.5, 1.4, 0.2))  # Example for Setosa
print(predict_species(6.0, 2.9, 4.5, 1.5))  # Example for Versicolor
print(predict_species(6.3, 3.3, 6.0, 2.5))  # Example for Virginica