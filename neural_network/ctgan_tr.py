
import pandas as pd

raw_data = pd.read_csv('/content/Chto-to_fixed.csv', sep=None, encoding='latin-1',engine='python')
raw_data.head()

raw_df = raw_data.copy()
raw_df.info()

raw_df.fillna(0)

raw_df = raw_df.transpose()

train_df = raw_df.select_dtypes(exclude="object")


# Выбираем столбцы с типом object
object_cols = raw_df.select_dtypes(exclude='float').columns.to_list()

# Заменяем ячейки на 0
for i in object_cols:
    raw_df[i] = raw_df[i].fillna(0)


"""#Нейросеть

1.   Новый пункт

1.   Новый пункт
2.   Новый пункт


2.   Новый пункт




"""

from ctgan import CTGAN

ctgan = CTGAN(epochs=6,pac=1)

categoricals = train_df.select_dtypes(include="object").columns.tolist()


ctgan.fit(train_df, categoricals)

synthetic_data = ctgan.sample(2000)
synthetic_data.head()

ctgan.save('my_model.pkl') #скачивание матрицы весов

ctgan = CTGAN.load('my_model.pkl') #подгрузка матрицы
#ctgan = CTGAN()
data = ctgan.sample(300) #генерация 300 сэмплов (300 строк с данными). ЧИСЛО ГЕНЕРАЦИЙ СТАВИТЬ КРАТНЫМ 3!!!


from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data=real_data,
    synthetic_data=synthetic_data,
    metadata=metadata)
