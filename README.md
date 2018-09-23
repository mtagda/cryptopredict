# cryptopredict
Prediction of Bitcoin price

## PROBLEM AND SOLUTION

The problem I am solving is to build a model to predict price (in USD) movement of Bitcoin (BTC).
In order to do that, you need to fetch historical data (here from the Cryptocompare API). I decided to fetch only the last 300 hourly data values. If I had more time I would test different number of fetched data.

    # Load the dataset
    endpoint = 'https://min-api.cryptocompare.com/data/histohour'
    res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=300&aggregate=1')

Then, we create a pandas DataFrame object to have a better access to our fetched data.

    hist = pandas.DataFrame(json.loads(res.content)['Data'])

Also, to use then some Machine Learning algorithm on our data, we need to reshape and normalize our dataset in range 0 to 1

    # Reshape dataset
    dataset = dataset.reshape(-1, 1)
    # Normalize dataset in range 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

The next step is to reshape dataset into X=t and Y=t+1 using the function prepare_dataset() as well as to split it into training and test data with split_dataset() in a very common proportion for ML algorithms 70% / 30%.

After all, we finally build a ML model using keras. To obtain a solution to our problem I decided to choose LSTM method as LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series.
If I had more time, I would work more on testing different activation and loss functions as well as optimizers for this model.
The model I decided to use is created as follows.

    model = Sequential()
    model.add(LSTM(32,
                   activation='tanh',
                   batch_input_shape=(batch_size, look_back, 1),
                   stateful=True,
                   return_sequences=False))
    model.add(Dense(activation="linear", units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

As next, we fit the model above using 20 epochs. Again, if I had more time I would focus more on choosing the right epochs parameter.

    model.fit(train_x, train_y, epochs=20, batch_size=batch_size, shuffle=False)

Finally, we are able to forecast future prices of Bitcoin calling the function make_forecast(). 
Below you can see a sample prediction for 6 next hours on 23-08-2018 12:34 PM EDT (orange) and initial dataset (blue)

![Plot](https://github.com/mtagda/cryptopredict/blob/master/figure.png)

## HOW TO RUN THE API
You just need to run the following commands

        docker build -t predictor .
        docker run -d -p localhost:5000 predictor
        curl http://localhost:5000/predict/BTC/USD

Sample output to the command above:
{"2018-09-23T16:00:00": "6498.334", "2018-09-23T17:00:00": "6434.0776", "2018-09-23T18:00:00": "6379.181", "2018-09-23T19:00:00": "6332.5576", "2018-09-23T20:00:00": "6293.8623", "2018-09-23T21:00:00": "6262.3164"}
