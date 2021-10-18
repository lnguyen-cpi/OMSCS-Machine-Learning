import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split

df = pd.read_csv("./dataset/occupancy/datatraining.txt",
                 # names=['date',
                 #        'Temperature',
                 #        'Humidity',
                 #        'Light',
                 #        'CO2',
                 #        'HumidityRatio',
                 #        'Occupancy'],
                 # header=True
                 )

f = '%Y-%m-%d %H:%M:%S'
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, f).toordinal())


data = df.loc[:, ['date', 'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']]
label = df.loc[:, ['Occupancy']]


training_data, test_data, training_label, test_label = train_test_split(
    data, label, test_size=0.75, random_state=42
)

