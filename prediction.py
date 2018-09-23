import numpy
import pandas
import matplotlib.pyplot as plt
import json
import requests
from tqdm import trange
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def get_data():
    """
    Loads historical BTC hourly prices.
    :return: tuple of dataset and the used MinMaxScaler
    """
    # Load the dataset
    endpoint = 'https://min-api.cryptocompare.com/data/histohour'
    res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=300&aggregate=1')
    hist = pandas.DataFrame(json.loads(res.content)['Data'])
    # Consider closing prices
    dataset = hist.close.values
    # Convert prices to float32
    dataset = dataset.astype('float32')
    # Reshape dataset
    dataset = dataset.reshape(-1, 1)
    # Normalize dataset in range 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    print(dataset)
    return dataset, scaler


def prepare_dataset(dataset, look_back=1):
    """
    Reshape dataset into X=t and Y=t+1
    :param dataset: numpy dataset
    :param look_back: number of previous time steps to consider
    :return: input and output dataset
    """
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        data_x.append(dataset[i:(i + look_back), 0])
        data_y.append(dataset[i + look_back, 0])
    return numpy.array(data_x), numpy.array(data_y)


def split_dataset(dataset, train_size, look_back):
    """
    Splits dataset into training and test datasets.
    :param dataset: source dataset
    :param train_size: specifies the train data size
    :param look_back: number of previous time steps as int
    :return: training and test dataset
    """
    if train_size < look_back:
        raise ValueError('train_size cannot be smaller than look_back')
    train, test = dataset[0:train_size, :], dataset[train_size - look_back:len(dataset), :]
    return train, test


def build_model(look_back, batch_size=1):
    """
    Builds a Sequential model
    :param look_back: number of previous time steps as int
    :param batch_size: batch_size as int, defaults to 1
    :return: keras Sequential model
    """
    model = Sequential()
    model.add(LSTM(64,
                   activation='relu',
                   batch_input_shape=(batch_size, look_back, 1),
                   stateful=True,
                   return_sequences=False))
    model.add(Dense(output_dim=1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def make_forecast(model, look_back_buffer, timesteps=1, batch_size=1):
    forecast_predict = numpy.empty((0, 1), dtype=numpy.float32)
    for _ in trange(timesteps, mininterval=1.0):
        # make prediction with current lookback buffer
        cur_predict = model.predict(look_back_buffer, batch_size)
        # add prediction to result
        forecast_predict = numpy.concatenate([forecast_predict, cur_predict], axis=0)
        # add new axis to prediction to make it suitable as input
        cur_predict = numpy.reshape(cur_predict, (cur_predict.shape[1], cur_predict.shape[0], 1))
        # remove oldest prediction from buffer
        look_back_buffer = numpy.delete(look_back_buffer, 0, axis=1)
        # concat buffer with newest prediction
        look_back_buffer = numpy.concatenate([look_back_buffer, cur_predict], axis=1)
    return forecast_predict


def main():
    dataset, scaler = get_data()

    # Split dataset into train and test sets
    look_back = int(len(dataset) * 0.20)
    train_size = int(len(dataset) * 0.70)
    train, test = split_dataset(dataset, train_size, look_back)

    # Reshape
    train_x, train_y = prepare_dataset(train, look_back)
    test_x, test_y = prepare_dataset(test, look_back)

    # Reshape input to get the form of [samples, time steps, features]
    train_x = numpy.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = numpy.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    # Create and fit model
    batch_size = 1
    model = build_model(look_back, batch_size=batch_size)
    model.fit(train_x, train_y, nb_epoch=50, batch_size=batch_size, shuffle=False)
    model.reset_states()

    # generate predictions for training
    train_predict = model.predict(train_x, batch_size)
    test_predict = model.predict(test_x, batch_size)

    # Generate future predictions for the next 6 hours
    future_predict = make_forecast(model, test_x[-1::], timesteps=6, batch_size=batch_size)
    # Invert dataset and predictions
    dataset = scaler.inverse_transform(dataset)

    future_predict = scaler.inverse_transform(future_predict)
    print(future_predict)

    plt.plot(dataset)
    plt.plot([None for _ in range(look_back)] +
             [None for _ in train_predict] +
             [None for _ in test_predict] +
             [x for x in future_predict])
    plt.show()


if __name__ == '__main__':
    main()
