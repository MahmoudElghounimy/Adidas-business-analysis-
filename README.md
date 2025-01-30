
### Analysis of Adidas Quarterly Sales Revenue

This repository contains a Python script that performs time series analysis on Adidas' quarterly sales data to predict future sales. The analysis uses various statistical models and visualizations to understand trends, seasonality, and patterns in the data. Below is a step-by-step breakdown of the methodology.

#### 1. **Loading the Data**
The dataset, `adidas-quarterly-sales.csv`, contains the quarterly sales revenue of Adidas. The data is loaded into a Pandas DataFrame for further processing.

```python
data = pd.read_csv("adidas-quarterly-sales.csv")
```

#### 2. **Visualizing the Sales Revenue**
A line plot is created using Plotly to visualize the quarterly sales revenue of Adidas. This allows for an initial understanding of the revenue trends over time.

```python
import plotly.express as px
figure = px.line(data, x="Time Period", y="Revenue", title='Quarterly Sales Revenue of Adidas in Millions')
figure.show()
```

#### 3. **Seasonal Decomposition**
The `seasonal_decompose` function from `statsmodels` is applied to decompose the sales revenue into three components: **Trend**, **Seasonality**, and **Residuals**. This decomposition helps in identifying underlying patterns in the data, such as seasonal fluctuations and long-term trends.

```python
result = seasonal_decompose(data["Revenue"], model='multiplicative', period=4)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(10, 15)
```

#### 4. **Autocorrelation and Partial Autocorrelation**
Autocorrelation and Partial Autocorrelation functions (ACF and PACF) are plotted to assess the relationships between lagged observations. These plots help in determining appropriate values for the ARIMA (AutoRegressive Integrated Moving Average) model parameters.

```python
pd.plotting.autocorrelation_plot(data["Revenue"])
plot_pacf(data["Revenue"], lags=20)
```

#### 5. **Building the SARIMAX Model**
A SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) model is built to forecast future sales revenue. The model is initialized with the parameters `p=1`, `d=1`, and `q=1` for both the non-seasonal and seasonal components. These values represent the autoregressive order, differencing, and moving average order, respectively. The model is then fitted to the data.

```python
model = sm.tsa.statespace.SARIMAX(data["Revenue"], order=(p, d, q), seasonal_order=(p, d, q, 12))
model = model.fit()
```

#### 6. **Model Summary**
After fitting the model, a summary of the SARIMAX model is printed. This includes important metrics like the model’s coefficients, the AIC (Akaike Information Criterion), and other diagnostic measures.

```python
print(model.summary())
```

#### 7. **Making Predictions**
The model is used to predict the next 7 quarters of Adidas' sales revenue. The predictions are made using the `predict()` method, starting from the last observed data point.

```python
predictions = model.predict(len(data), len(data) + 7)
```

#### 8. **Plotting the Predictions**
The historical sales data and the predicted sales revenue are plotted together for visual comparison. The training data is displayed alongside the predicted values to assess the accuracy of the model.

```python
data["Revenue"].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")
```
