import pandas as pd
import numpy as np

df = pd.read_csv('data/train.csv')
df
price_mean = df['SalePrice'].mean()
price_mean

sample = pd.read_csv('data/sample_submission.csv')
sample.shape
sample['SalePrice'] = price_mean
sample
sample = sample.to_csv('data/sample_submission.csv', index = False)
