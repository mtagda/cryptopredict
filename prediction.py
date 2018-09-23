import numpy
import pandas
import json
import requests
from tqdm import trange
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import datetime


class DataProcessing(object):
    @staticmethod
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
        # Get the latest date and time
        last_timestamp = hist.time.values[-1]
        last_datetime = datetime.datetime.fromtimestamp(last_timestamp)
        # Convert prices to float32
        dataset = dataset.astype('float32')
        # Reshape dataset
        dataset = dataset.reshape(-1, 1)
        # Normalize dataset in range 0-1
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        return dataset, scaler, last_datetime

    @staticmethod
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

    @staticmethod
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


class Forecasting(object):
    def __init__(self, batch_size=1):
        # batch_size default 1
        self.batch_size = batch_size

    def build_model(self, look_back):
        """
        Builds a Sequential model
        :param look_back: number of previous time steps as int
        :return: keras Sequential model
        """
        model = Sequential()
        model.add(LSTM(32,
                       activation='tanh',
                       batch_input_shape=(self.batch_size, look_back, 1),
                       stateful=True,
                       return_sequences=False))
        model.add(Dense(activation="linear", units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def make_forecast(self, model, look_back_buffer, timesteps=1):
        """
        Make forecast using previously created model
        :param model: keras Sequential model
        :param look_back_buffer: previous look_back data
        :param timesteps: number of data do be predicted
        :return: predicted values
        """
        forecast_predict = numpy.empty((0, 1), dtype=numpy.float32)
        for _ in trange(timesteps, mininterval=1.0):
            # make prediction with current lookback buffer
            cur_predict = model.predict(look_back_buffer, self.batch_size)
            # add prediction to result
            forecast_predict = numpy.concatenate([forecast_predict, cur_predict], axis=0)
            # add new axis to prediction to make it suitable as input
            cur_predict = numpy.reshape(cur_predict, (cur_predict.shape[1], cur_predict.shape[0], 1))
            # remove oldest prediction from buffer
            look_back_buffer = numpy.delete(look_back_buffer, 0, axis=1)
            # concat buffer with newest prediction
            look_back_buffer = numpy.concatenate([look_back_buffer, cur_predict], axis=1)
        return forecast_predict


def generate_dates_hourly(start, n):
    """
    Generates future n dates with hourly timestep with respect to start date
    :param start: start datetime
    :param number: number of future dates to be created
    :return: future n dates, start not included
    """
    future_datetimes = []
    for t in range(1, n + 1):
        future_datetimes.append(start + datetime.timedelta(hours=t))
    return future_datetimes


def predict():
    dataset, scaler, last_datetime = DataProcessing.get_data()

    # Split dataset into train and test sets
    look_back = int(len(dataset) * 0.20)
    train_size = int(len(dataset) * 0.70)
    train, test = DataProcessing.split_dataset(dataset, train_size, look_back)

    # Reshape
    train_x, train_y = DataProcessing.prepare_dataset(train, look_back)
    test_x, test_y = DataProcessing.prepare_dataset(test, look_back)

    # Reshape input to get the form of [samples, time steps, features]
    train_x = numpy.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = numpy.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    # Create and fit model
    batch_size = 1
    Forecast = Forecasting(batch_size)
    model = Forecast.build_model(look_back)
    model.fit(train_x, train_y, epochs=20, batch_size=batch_size, shuffle=False)
    model.reset_states()

    # Generate future predictions for the next 6 hours
    future_predict = Forecast.make_forecast(model, test_x[-1::], timesteps=6)

    # Invert predictions
    future_predict = scaler.inverse_transform(future_predict)

    # Get future datetimes
    future_dates = generate_dates_hourly(last_datetime, 6)

    # Create JSON output of the form {date: price}
    dict_output = {future_dates[i].isoformat(): str(future_predict[i][0]) for i in range(0, 6)}
    json_output = json.dumps(dict_output)

    print(json_output)
    return json_output


predict()
