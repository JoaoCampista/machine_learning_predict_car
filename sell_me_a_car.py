#%% Importando as bibliotecas

import pandas as pd
import numpy as np
import graphviz
import pydotplus

from IPython.display import Image  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score

from datetime import datetime

#%% Importando a base
uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
raw_database = pd.read_csv(uri, sep = ',', low_memory = False)

#%% Trabalhando a base

ano_atual = datetime.today().year
mi_to_km = 1.60934

a_trocar = {
    'yes' : 1,
    'no' : 0}

a_renomear = {'mileage_per_year' : 'milhas_por_ano',
              'model_year' : 'ano_do_modelo',
              'price' : 'preco',
              'sold' : 'vendido'}

a_remover = ['Unnamed: 0',
             'milhas_por_ano',
             'ano_do_modelo']

raw_database = raw_database.rename(columns = a_renomear)
raw_database['vendido'] = raw_database['vendido'].map(a_trocar)
raw_database['km_por_ano'] = raw_database['milhas_por_ano'] * mi_to_km
raw_database['idade_do_modelo'] = ano_atual - raw_database['ano_do_modelo']

raw_database = raw_database.drop(columns = a_remover, axis = 1)

del(a_remover, a_renomear, a_trocar, ano_atual, mi_to_km, uri)

#%% Separação das bases

X_colunas = ['idade_do_modelo',
             'km_por_ano',
             'preco',]

y_colunas = ['vendido',]

X = raw_database[X_colunas]
y = raw_database[y_colunas]

#%% Divisao dos dados de treino e teste

SEED = 23
np.random.seed(SEED)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = SEED,
                                                    stratify = y)
'''
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

'''

X_train = X_train_raw
X_test = X_test_raw

print('y_test com proporção de %.2f valores positivos' % y_test['vendido'].value_counts(normalize = True)[1])
print('y_train com proporção de %.2f valores positivos' % y_train['vendido'].value_counts(normalize = True)[1])


#%% Dummy predict

# dummy stratified
dummy_stratified = DummyClassifier(strategy = 'stratified',random_state = SEED)
dummy_stratified.fit(X_train, y_train.values.ravel())
y_pred_stratified = dummy_stratified.predict(X_test)

# dummy most_frequent
dummy_most_frequent = DummyClassifier(strategy = 'most_frequent',random_state = SEED)
dummy_most_frequent.fit(X_train, y_train.values.ravel())
y_pred_most_frequent = dummy_most_frequent.predict(X_test)

#%% Regressão dos dados

model =  DecisionTreeClassifier(random_state = SEED, max_depth = 4)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)

#%% Verificação da regressão

accuracy_model = accuracy_score(y_test, y_pred, normalize = True)*100
accuracy_dummy_stratified = accuracy_score(y_test, y_pred_stratified, normalize = True)*100
accuracy_dummy_most_frequent = accuracy_score(y_test, y_pred_most_frequent, normalize = True)*100

verificador = pd.DataFrame()

verificador['y_pred'] = y_pred
verificador['y_test'] = y_test.reset_index(drop=True)

print("Modelo treinado com %d elementos e testado com %d elementos" % (len(X_train), len(X_test)))
print('A precisão do dummy stratified foi de %.2f%%' % (accuracy_dummy_stratified))
print('A precisão do dummy most_frequent foi de %.2f%%' % (accuracy_dummy_most_frequent))
print('A precisão do modelo foi de %.2f%%' % (accuracy_model))

#%% Visualização dos resultados

dot_data = export_graphviz(model, out_file=None,
                           filled = True, rounded = True,
                           feature_names = X_colunas,
                           class_names = ['não', 'sim'])


#grafico = graphviz.Source(dot_data)

grafico = pydotplus.graph_from_dot_data(dot_data)
Image(grafico.create_png())

grafico.write_png("car_decisiontree.png")
