import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {'Age': [25, 30, 30, 30, 45, 50, 50, 50, 50, 50]}
data_frame = pd.DataFrame(data)

# Visualize the distribution of the 'Age' column
data_frame.hist(column='Age', bins=5, grid=False, color='blue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
