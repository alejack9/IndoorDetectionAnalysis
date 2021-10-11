import os
import json
import pandas as pd
from pandas.io.json import json_normalize

data = pd.DataFrame()
for file in os.listdir('./jsons'):
    with open('./jsons/' + file) as f:
        d = json.load(f)

    # print('./jsons/' + file)
    data = pd.concat([data, pd.DataFrame(json_normalize(d))])
data.to_csv('signalfingerprints.csv')
print(f"Columns number: {len(data.columns)}")
