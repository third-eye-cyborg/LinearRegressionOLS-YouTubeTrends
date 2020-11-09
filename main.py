import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# load data
data = pd.read_csv('archive/USvideos.csv')

# break data down for analysis
df = data[['title', 'views', 'likes', 'dislikes', 'comment_count']]
views = data[['title', 'views']]
likes = data['likes']
dislikes = data['dislikes']
comment_count = data['comment_count']

# create feature list
train_list = [likes, dislikes, comment_count]

# print head to important data
df.head()

# create scaler variable
scaler = StandardScaler()

# get feature titles ready
labels = ['likes', 'dislikes', 'comment_count']

# get y ready and preprocessed
y = views['views']
y_scaled = y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y_scaled)

# get x ready and preprocessed
for i, x in enumerate(train_list):
    x_scaled = x.values.reshape(-1, 1)
    x_scaled = scaler.fit_transform(x_scaled)

    # split data for fitting and predicting with the model
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)

    # create model
    reg = linear_model.LinearRegression()

    # fit model
    reg.fit(X_train, y_train)

    # make prediction
    y_pred = reg.predict(X_test)

    # check the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # check the score function
    r2s = r2_score(y_test, y_pred)

    # print feature labels
    print(labels[i])

    # print mse
    print(f'Mean squared error: {mse}')

    # 1 equals perfect prediction
    print(f'Coefficient of determination: {r2s}')


