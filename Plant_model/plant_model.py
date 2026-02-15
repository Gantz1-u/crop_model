import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = np.genfromtxt('plant_data.csv', delimiter=',', 
                     dtype=None, names=True, encoding='utf-8')

# Encode Soil Type (categorical feature)
le_soil = LabelEncoder()
soil_encoded = le_soil.fit_transform(data['Soil_Type'])

# Prepare features
X = np.column_stack([
    data['Temparature'],
    data['Humidity'],
    data['Moisture'],
    soil_encoded
])

# Encode target
le_crop = LabelEncoder()
y = le_crop.fit_transform(data['Crop_Type'])


