import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../dataset/car_valuation/car.data",
                 names=['buying_price',
                        'maintained_price',
                        'doors', 'persons',
                        'lug_boot',
                        'safety',
                        'label'],
                 header=None)

# Remove all labels 'good' and 'vgood'
# df = df[df['label'] != 'good']
# df = df[df['label'] != 'vgood']

cleanup_nums = {
    "buying_price": {
        "vhigh": 3, "high": 2, "med": 1, "low": 0,
    },
    "maintained_price": {
        "low": 0, "med": 1, "high": 2, "vhigh": 3,
    },
    "doors": {
        "5more": 5
    },
    "persons": {
        "more": 5
    },
    "lug_boot": {
        "small": 0, "med": 1, "big": 2,
    },
    "safety": {
        "low": 0, "med": 1, "high": 2
    },
    "label": {
        "unacc": 0, "acc": 1,
        "good": 2, "vgood": 3
    }

}

df = df.replace(cleanup_nums)

df = df.astype(int)


data = df.loc[:, ['buying_price', 'maintained_price', 'doors', 'persons', 'lug_boot', 'safety']]
label = df.loc[:, ['label']]


training_data, test_data, training_label, test_label = train_test_split(
    data, label, test_size=0.75, random_state=42
)

