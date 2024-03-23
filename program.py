import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych z pliku CSV
data = pd.read_csv('shootouts.csv')

# Wyświetlenie kilku pierwszych wierszy danych
print(data.head())

# Informacje o danych
print(data.info())

# Sprawdzenie brakujących wartości
print(data.isnull().sum())

print("\n\n")
print(data.columns)
print("\n\n")

# Przypisanie unikalnych identyfikatorów dla drużyn
all_teams = set(data['home_team'].unique()).union(set(data['away_team'].unique()))
team_to_id = {team: i for i, team in enumerate(all_teams)}

# Zastąpienie nazw drużyn ich ID w danych
data['home_team_id'] = data['home_team'].map(team_to_id)
data['away_team_id'] = data['away_team'].map(team_to_id)

# Mapowanie zwycięskich drużyn na ich identyfikatory
data['winner_id'] = data['winner'].map(team_to_id)

# Wyświetlenie danych z nowymi kolumnami zawierającymi ID drużyn
print(data.head())

# Rozpakowanie danych
teams = list(team_to_id.keys())
team_ids = list(team_to_id.values())

team_id_df = pd.DataFrame({'Drużyna': teams, 'ID drużyny': team_ids})

# Zapisanie do pliku CSV
team_id_df.to_csv('teams_and_ids.csv', index=False)

# Wizualizacja rozkładu danych
plt.figure(figsize=(10, 6))
plt.hist(data['winner_id'])
plt.xlabel('Zwycięzca meczu')
plt.ylabel('Liczba meczów')
plt.title('Rozkład zwycięzców meczów')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Podział danych na zbiór treningowy i testowy
X = data[['home_team_id', 'away_team_id']]  # cechy
y = data['winner_id']  # etykiety
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definicja i uczenie modeli
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Ocena modelu
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print()

