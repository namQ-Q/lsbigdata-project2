import pandas as pd
import numpy as np
df_exam = pd.read_excel('data/excel_exam.xlsx')
sum(df_exam['math'])/20
sum(df_exam['english'])/20
sum(df_exam['science'])/20

df_exam.shape
len(df_exam)
df_exam.size

# df_exam = pd.read_excel('data/excel_exam.xlsx',
#                         sheet_name = "sheet2") # sheet2 가져오기

df_exam['total'] = df_exam['math'] + df_exam['english'] + df_exam['science']
df_exam['mean'] = (df_exam['math'] + df_exam['english'] + df_exam['science']) / 3
df_exam

df_exam[(df_exam['math'] > 50) & (df_exam['english'] > 50)]
df_nc3 = df_exam[df_exam['nclass'] == 3]
df_nc3[['math', 'english', 'science']]
df_nc3[0:1]
df_nc3

df_exam[7:16:2]

df_exam.sort_values(['nclass','math'], ascending = [True, False])


np.where(a>3, 'up', 'down')
df_exam['updown'] = np.where(df_exam['math'] > 50, 'up', 'down')
df_exam









