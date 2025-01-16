import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate, cross_val_predict, KFold
#from sklearn.metrics import mean_squared_error
import xgboost as xgb
import time
import matplotlib.pyplot as plt

start_time = time.time()

# データの読み込み
d_train = pd.read_csv("./data/train.csv")
d_test = pd.read_csv("./data/test.csv")

# 訓練データの欠損値のある行を削除
d_train = d_train.dropna(subset=['popularity'])

# ---------- データの前処理 ----------
print("データの前処理を行います")
d_train.fillna(0, inplace=True)
d_test.fillna(0, inplace=True)

# 訓練データとテストデータの結合
d_test['popularity'] = -1  # ダミーの値を追加
combined_data = pd.concat([d_train, d_test], ignore_index=True)

# カテゴリカルデータの変換とエンコード(Label Encoding)
label_encoder = LabelEncoder()
for col in ['artists', 'album_name', 'track_name', 'track_genre']:
    combined_data[col] = combined_data[col].astype(str)
    combined_data[f'{col}_label'] = label_encoder.fit_transform(combined_data[col])

# カテゴリカルデータの変換とエンコード(One-hot Encoding)(元のデータ列は消える)
combined_data = combined_data.astype({'explicit': int}) # `explicit`の列の型をintにキャスト
categorical_columns = ['track_genre'] # One-hotが効果ありそうなtrack_genreだけ
combined_data = pd.get_dummies(combined_data, columns=categorical_columns, dtype=int) # get_dummiesを使ってone-hotエンコーディング．columnsに指定された列のみone-hotエンコーディングし，出力としてint型を指定

# ---------- 新しい特徴量の作成 ----------
print("新たな特徴量を作成します")
combined_data['track_name_length'] = combined_data['track_name'].astype(str).apply(len)
combined_data['num_artists'] = combined_data['artists'].astype(str).apply(lambda x: x.count(';') + 1)
combined_data['tempo_loudness_interaction'] = combined_data['tempo'] * combined_data['loudness']

# トラックの長さカテゴリ
combined_data['duration_category'] = pd.cut(combined_data['duration_ms'], bins=[0, 180000, 300000, 600000], labels=['short', 'medium', 'long'])
# duration_categoryをone-hotエンコーディングに変換
duration_dummies = pd.get_dummies(combined_data['duration_category'], prefix='duration')
combined_data = pd.concat([combined_data, duration_dummies], axis=1)
combined_data.drop(columns=['duration_category'], inplace=True)

# エネルギーとダンス適性の積
combined_data['energy_danceability_interaction'] = combined_data['energy'] * combined_data['danceability']

# 素のカテゴリカル変数によるデータを削除(アーティスト名は後で使う、ジャンル名はOne-hot Encoding時に削除済み)
combined_data.drop(columns=['album_name', 'track_name'], inplace=True)

# 訓練データとテストデータに分割
d_train = combined_data[combined_data['popularity'] != -1].copy()
d_test = combined_data[combined_data['popularity'] == -1].copy()
d_test.drop(columns=['popularity'], inplace=True)

# ---------- 特徴量エンジニアリング ----------
print("特徴量エンジニアリングを行います")

# アーティスト名を分割
d_train['artists_list'] = d_train['artists'].astype(str).str.split(';')
d_test['artists_list'] = d_test['artists'].astype(str).str.split(';')

# 以下、平均人気度によるカテゴリカル変数のTarget Encodingを交差検証しながら行う
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# アルバムごと、（曲名ごと？、）ジャンルごとに計算
for col in ['album_name_label', 'track_name_label', 'track_genre_label']:
    d_train[f'{col}_popularity'] = np.nan
    for train_index, val_index in kf.split(d_train): # 訓練データとテストデータとなるデータのインデックス列をそれぞれ取得
        X_train_fold = d_train.iloc[train_index]
        mean_target = X_train_fold.groupby(col)['popularity'].mean() # 訓練データ内でカテゴリカル変数ごとの平均値を計算
        d_train.loc[d_train.index[val_index], f'{col}_popularity'] = d_train.loc[d_train.index[val_index], col].map(mean_target) # テストデータに適用

    # テストデータに対しても平均値でEncoding
    global_mean_target = d_train.groupby(col)['popularity'].mean()
    d_test[f'{col}_popularity'] = d_test[col].map(global_mean_target)

# アーティストごとに計算
d_train['artists_popularity'] = np.nan
for train_index, val_index in kf.split(d_train):
    X_train_fold = d_train.iloc[train_index]
    artist_popularity = X_train_fold.explode('artists_list').groupby('artists_list')['popularity'].mean()
    d_train.loc[val_index, 'artists_popularity'] = d_train.loc[val_index, 'artists_list'].apply(lambda x: np.mean([artist_popularity.get(artist, np.nan) for artist in x]))

# テストデータもEncoding
global_artist_popularity = d_train.explode('artists_list').groupby('artists_list')['popularity'].mean()
d_test['artists_popularity'] = d_test['artists_list'].apply(lambda x: np.mean([global_artist_popularity.get(artist, np.nan) for artist in x]))

# Target Encodingの後にartists_list列を削除(そのまま使うと型エラー)
d_train.drop(columns=['artists', 'artists_list'], inplace=True)
d_test.drop(columns=['artists', 'artists_list'], inplace=True)

# スケーリング
scaler = StandardScaler()
numeric_features = [col for col in d_train.select_dtypes(include=[np.number]).columns if col != 'popularity']
d_train[numeric_features] = scaler.fit_transform(d_train[numeric_features])
d_test[numeric_features] = scaler.transform(d_test[numeric_features])

# 訓練データの特徴量とターゲット(popularity)の分割
X_train = d_train.drop(columns=['track_id', 'popularity'])
y_train = d_train['popularity']
X_test = d_test.drop(columns=['track_id']) # ついでにテストデータも加工

# XGBoostモデルのインスタンスを作成
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# ---------- ハイパーパラメータのランダムサーチ ----------
print("ハイパーパラメータのランダムサーチを行います")
param_dist = {
    'max_depth': [11],
    'learning_rate': [0.0485],
    'n_estimators':[800],
    'subsample': [0.8],
    'colsample_bytree': [1.0]
    # 'max_depth': [9, 11, 13],
    # 'learning_rate': [0.03, 0.05, 0.1, 0.5],
    # 'n_estimators': [800, 1000, 1200],
    # 'subsample': [0.8, 0.9, 1.0],
    # 'colsample_bytree': [0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, 
                                   scoring='neg_mean_squared_error', n_iter=5, cv=5, verbose=1, n_jobs=-1, random_state=42, return_train_score=True)
random_search.fit(X_train, y_train, verbose=True)

# 最適なモデルの取得
best_xgb_model = random_search.best_estimator_
print(f'Best parameters found by random search are: {random_search.best_params_}')

# 交叉検証の結果を表示
cv_results = random_search.cv_results_
#cv_results = cross_validate(best_xgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', return_train_score=True) # (RandomizedSearchCV内で交叉検証済みなので、再度計算して同じ結果が返るだけ)
best_index = random_search.best_index_
print("Cross-validation results:")
print(f"mean train score: {-cv_results['mean_train_score'][best_index]}")
print(f"mean test score: {-cv_results['mean_test_score'][best_index]}") # random_search.best_score_と同一

# テストデータの予測
y_test_pred = best_xgb_model.predict(X_test)

# 予測値を0から100の範囲に制限
y_test_pred = np.clip(y_test_pred, 0, 100)

# 結果の保存（popularityのみの場合）
submission = pd.DataFrame(y_test_pred, columns=['popularity'])
submission.to_csv('y_pred_XGB.csv', index=False, header=False)
print('結果保存完了')

# 実行時間表示
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time} seconds")
