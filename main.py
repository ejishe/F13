import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import os
import joblib

file_path = 'payments.xlsx'
prepared_data_file = 'prepared_data.csv'
model_file = 'donation_model.pkl'

# Загрузка или подготовка данных
if os.path.exists(prepared_data_file):
    data = pd.read_csv(prepared_data_file)
else:
    payments_df = pd.read_excel(file_path)
    payments_df['finished_at'] = pd.to_datetime(payments_df['finished_at'], format='%d.%m.%Y %H:%M:%S')
    payments_df['amount'] = pd.to_numeric(payments_df['amount'])
    user_donations = payments_df.groupby('user_id')['amount'].agg(['sum', 'count']).reset_index()
    user_donations.columns = ['user_id', 'total_donated', 'donation_count']
    payments_df['day_of_week'] = payments_df['finished_at'].dt.dayofweek
    payments_df['month'] = payments_df['finished_at'].dt.month
    payments_df['hour'] = payments_df['finished_at'].dt.hour
    data = pd.merge(payments_df[['user_id', 'amount', 'day_of_week', 'month', 'hour']], user_donations, on='user_id')
    data.to_csv(prepared_data_file, index=False)
X = data[['total_donated', 'donation_count', 'day_of_week', 'month', 'hour']]
y = data['amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if os.path.exists(model_file):
    model = joblib.load(model_file)
else:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, model_file)
def recommend_donation_amount(user_id, data, model, current_time):
    user_data = data[data['user_id'] == user_id]
    if not user_data.empty:
        day_of_week = current_time.dayofweek
        month = current_time.month
        hour = current_time.hour

        user_features = user_data[['total_donated', 'donation_count']].copy()
        user_features['day_of_week'] = day_of_week
        user_features['month'] = month
        user_features['hour'] = hour

        recommended_amount = model.predict(user_features)
        recommended_amount_rounded = np.ceil(recommended_amount / 10) * 10
        return recommended_amount_rounded[0]
    return None

user_id_to_test = '44eb4f57-d469-5fb7-f6f0-90ab52058b89'
current_time = pd.Timestamp.now()
recommended_amount = recommend_donation_amount(user_id_to_test, data, model, current_time)

print(f'Рекомендуемая сумма пожертвования для пользователя с user_id {user_id_to_test}: {recommended_amount}')
