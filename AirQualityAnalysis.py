import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import TimeSeriesSplit


df = pd.read_csv('AirQualityUCI.csv', header=0)

# Display the first few rows of the dataset
print("First 5 Rows ")
print(df.head())

# Quick plot of the dataset structure
df.plot(subplots=True, figsize=(10, 8))
pyplot.suptitle("Air Quality Features Overview")
pyplot.show()

# Extract values for modeling

X = df.select_dtypes(include=['float64', 'int64']).values

# 2. Simple Train-Test Split for Time Series
train_size = int(len(X) * 0.66)
train, test = X[0:train_size], X[train_size:len(X)]

print('\n--- Simple Train-Test Split ---')
print('Total Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))

# Plot simple train-test split
pyplot.figure(figsize=(10, 4))
pyplot.plot(train[:, 0], label='Train')
pyplot.plot([None for i in train] + [x[0] for x in test], label='Test')
pyplot.title("Simple Train-Test Split (First Numeric Feature)")
pyplot.legend()
pyplot.show()

# 3. Multiple Train-Test Splits using TimeSeriesSplit
print('Time Series Split Cross-Validation')
tscv = TimeSeriesSplit(n_splits=3)
pyplot.figure(figsize=(10, 8))
index = 1

for train_index, test_index in tscv.split(X):
    train_fold = X[train_index]
    test_fold = X[test_index]

    print(f'Split {index} -> Train Obs: {len(train_fold)}, Test Obs: {len(test_fold)}')

    pyplot.subplot(3, 1, index)
    pyplot.plot(train_fold[:, 0], label='Train')
    pyplot.plot([None for i in train_fold] + [x[0] for x in test_fold], label='Test')
    pyplot.title(f"TimeSeriesSplit Fold {index}")
    index += 1

pyplot.tight_layout()
pyplot.show()

# 4. Walk Forward Validation Setup
print('Walk-Forward Validation Structure Ready')
n_train = int(len(X) * 0.5)  # Initial training window size
n_records = len(X)

