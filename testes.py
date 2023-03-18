import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Exemplo Dataframe 
df = pd.DataFrame({'col_A':[1,5,7,8],'col_B':[9,7,4,3]})

# Salvar dataframe como arquivo csv na pasta atual
df.to_csv('simu-tempo-deslocamento_mun_T.csv', index = False, encoding='utf-8') # False: not include index
print(df)

# Carrega os dados
data = pd.read_csv('simu-tempo-deslocamento_mun_T.csv')

# Separa os dados em features e labels
if 'col_A' in data.columns:
    X = data.drop('col_A', axis=1)
    y = data['col_A']
else:
    X = data.drop('col_B', axis=1)
    y = data['col_B']

# Divide os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cria o modelo
model = LogisticRegression()

# Treina o modelo
model.fit(X_train, y_train)

# Faz as previsões
y_pred = model.predict(X_test)

# Calcula a acurácia
accuracy = accuracy_score(y_test, y_pred)

print('Acurácia:', accuracy)
