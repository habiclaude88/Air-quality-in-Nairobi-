# Air-quality-in-Nairobi-
import warnings
​
import wqet_grader
​
warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 3 Assessment")
# Import libraries here
import inspect
import time
import warnings
​
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from IPython.display import VimeoVideo
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
1. Prepare Data
1.1. Connect
Task 3.5.1: Connect to MongoDB server running at host "localhost" on port 27017. Then connect to the "air-quality" database and assign the collection for Dar es Salaam to the variable name dar.

client = MongoClient(host = 'localhost', port = 27017)
db = client['air-quality']
dar = db['dar-es-salaam']
wqet_grader.grade("Project 3 Assessment", "Task 3.5.1", [dar.name])
You got it. Dance party time! 🕺💃🕺💃

Score: 1

1.2. Explore
Task 3.5.2: Determine the numbers assigned to all the sensor sites in the Dar es Salaam collection. Your submission should be a list of integers.

sites = dar.distinct('metadata.site')
sites
[11, 23]
wqet_grader.grade("Project 3 Assessment", "Task 3.5.2", sites)
Very impressive.

Score: 1

Task 3.5.3: Determine which site in the Dar es Salaam collection has the most sensor readings (of any type, not just PM2.5 readings). You submission readings_per_site should be a list of dictionaries that follows this format:

[{'_id': 6, 'count': 70360}, {'_id': 29, 'count': 131852}]
Note that the values here ☝️ are from the Nairobi collection, so your values will look different.

result = dar.aggregate(
    [
        {'$group':{'_id': '$metadata.site', 'count': {'$count':{}}}}
    ]
)
readings_per_site = list(result)
readings_per_site
[{'_id': 11, 'count': 138412}, {'_id': 23, 'count': 60020}]
wqet_grader.grade("Project 3 Assessment", "Task 3.5.3", readings_per_site)
You got it. Dance party time! 🕺💃🕺💃

Score: 1

1.3. Import
Task 3.5.4: (5 points) Create a wrangle function that will extract the PM2.5 readings from the site that has the most total readings in the Dar es Salaam collection. Your function should do the following steps:

Localize reading time stamps to the timezone for "Africa/Dar_es_Salaam".
Remove all outlier PM2.5 readings that are above 100.
Resample the data to provide the mean PM2.5 reading for each hour.
Impute any missing values using the forward-will method.
Return a Series y.
def wrangle(collection):
    results = collection.find({"metadata.site": 11, "metadata.measurement": "P2"},
                             projection={"P2": 1, "timestamp": 1, "_id": 0},)
    
    y = pd.DataFrame(results).set_index('timestamp')
    
    # localize timezone
    y.index = y.index.tz_localize('UTC').tz_convert("Africa/Dar_es_Salaam")
    
    # remove outliers
    y = y[y['P2'] < 100]
    # filling NaN with ffill
    
    y = y['P2'].resample('1H').mean().fillna(method = 'ffill')
    
    return y
Use your wrangle function to query the dar collection and return your cleaned results.

y = wrangle(dar)
y.head()
timestamp
2018-01-01 03:00:00+03:00    9.456327
2018-01-01 04:00:00+03:00    9.400833
2018-01-01 05:00:00+03:00    9.331458
2018-01-01 06:00:00+03:00    9.528776
2018-01-01 07:00:00+03:00    8.861250
Freq: H, Name: P2, dtype: float64
​
wqet_grader.grade("Project 3 Assessment", "Task 3.5.4", wrangle(dar))
Yes! Great problem solving.

Score: 1

1.4. Explore Some More
Task 3.5.5: Create a time series plot of the readings in y. Label your x-axis "Date" and your y-axis "PM2.5 Level". Use the title "Dar es Salaam PM2.5 Levels".

fig, ax = plt.subplots(figsize=(15, 6))
y.plot(xlabel = "Date", ylabel = "PM2.5 Level", title = "Dar es Salaam PM2.5 Levels",ax=ax);
# Don't delete the code below 👇
plt.savefig("images/3-5-5.png", dpi=150)
​

with open("images/3-5-5.png", "rb") as file:
    wqet_grader.grade("Project 3 Assessment", "Task 3.5.5", file)
Python master 😁

Score: 1

Task 3.5.6: Plot the rolling average of the readings in y. Use a window size of 168 (the number of hours in a week). Label your x-axis "Date" and your y-axis "PM2.5 Level". Use the title "Dar es Salaam PM2.5 Levels, 7-Day Rolling Average".

fig, ax = plt.subplots(figsize=(15, 6))
y.rolling(168).mean().plot(ax=ax, xlabel = "Date", ylabel='PM2.5 Level', title="Dar es Salaam PM2.5 Levels, 7-Day Rolling Average");
# Don't delete the code below 👇
plt.savefig("images/3-5-6.png", dpi=150)
​

with open("images/3-5-6.png", "rb") as file:
    wqet_grader.grade("Project 3 Assessment", "Task 3.5.6", file)
Awesome work.

Score: 1

Task 3.5.7: Create an ACF plot for the data in y. Be sure to label the x-axis as "Lag [hours]" and the y-axis as "Correlation Coefficient". Use the title "Dar es Salaam PM2.5 Readings, ACF".

fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Dar es Salaam PM2.5 Readings, ACF");
# Don't delete the code below 👇
plt.savefig("images/3-5-7.png", dpi=150)
​

with open("images/3-5-7.png", "rb") as file:
    wqet_grader.grade("Project 3 Assessment", "Task 3.5.7", file)
Good work!

Score: 1

Task 3.5.8: Create an PACF plot for the data in y. Be sure to label the x-axis as "Lag [hours]" and the y-axis as "Correlation Coefficient". Use the title "Dar es Salaam PM2.5 Readings, PACF".

fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Dar es Salaam PM2.5 Readings, PACF");
# Don't delete the code below 👇
plt.savefig("images/3-5-8.png", dpi=150)
​

with open("images/3-5-8.png", "rb") as file:
    wqet_grader.grade("Project 3 Assessment", "Task 3.5.8", file)
Wow, you're making great progress.

Score: 1

1.5. Split
int(len(y) * 0.9)
1533
Task 3.5.9: Split y into training and test sets. The first 90% of the data should be in your training set. The remaining 10% should be in the test set.

cutoff_test = int(len(y) * 0.90)
y_train = y.iloc[:cutoff_test]
y_test = y.iloc[cutoff_test:]
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
y_train shape: (1533,)
y_test shape: (171,)
​
wqet_grader.grade("Project 3 Assessment", "Task 3.5.9a", y_train)
Yes! Great problem solving.

Score: 1

​
wqet_grader.grade("Project 3 Assessment", "Task 3.5.9b", y_test)
You = coding 🥷

Score: 1

2. Build Model
2.1. Baseline
Task 3.5.10: Establish the baseline mean absolute error for your model.

y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
​
print("Mean P2 Reading:", y_train_mean)
print("Baseline MAE:", mae_baseline)
Mean P2 Reading: 8.617582545265433
Baseline MAE: 4.07658759405218
wqet_grader.grade("Project 3 Assessment", "Task 3.5.10", [mae_baseline])
Yes! Your hard work is paying off.

Score: 1

2.2. Iterate
Task 3.5.11: You're going to use an AR model to predict PM2.5 readings, but which hyperparameter settings will give you the best performance? Use a for loop to train your AR model on using settings for p from 1 to 30. Each time you train a new model, calculate its mean absolute error and append the result to the list maes. Then store your results in the Series mae_series.

p_params = range(1, 31)
maes = []
​
for p in p_params:
    # Note start time
    start_time = time.time()
    # Train model
    model = AutoReg(y_train, lags = p).fit()
    # Calculate model training time
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Trained AR {p} in {elapsed_time} seconds.")
    # Generate in-sample (training) predictions
    y_pred = model.predict()
     # Calculate training MAE
    mae = mean_absolute_error(y_train.iloc[p:], y_pred.iloc[p:])
    # Append MAE to list in dictionary
    maes.append(mae)
​
    
    
mae_series = pd.Series(maes, name="mae", index=p_params)
mae_series.head()
Trained AR 1 in 0.01 seconds.
Trained AR 2 in 0.0 seconds.
Trained AR 3 in 0.0 seconds.
Trained AR 4 in 0.0 seconds.
Trained AR 5 in 0.0 seconds.
Trained AR 6 in 0.01 seconds.
Trained AR 7 in 0.11 seconds.
Trained AR 8 in 0.29 seconds.
Trained AR 9 in 0.1 seconds.
Trained AR 10 in 0.0 seconds.
Trained AR 11 in 0.1 seconds.
Trained AR 12 in 0.01 seconds.
Trained AR 13 in 0.1 seconds.
Trained AR 14 in 0.2 seconds.
Trained AR 15 in 0.1 seconds.
Trained AR 16 in 0.01 seconds.
Trained AR 17 in 0.01 seconds.
Trained AR 18 in 0.1 seconds.
Trained AR 19 in 0.01 seconds.
Trained AR 20 in 0.01 seconds.
Trained AR 21 in 0.1 seconds.
Trained AR 22 in 0.19 seconds.
Trained AR 23 in 0.2 seconds.
Trained AR 24 in 0.1 seconds.
Trained AR 25 in 0.3 seconds.
Trained AR 26 in 0.5 seconds.
Trained AR 27 in 0.3 seconds.
Trained AR 28 in 0.4 seconds.
Trained AR 29 in 0.5 seconds.
Trained AR 30 in 0.7 seconds.
1    0.947888
2    0.933894
3    0.920850
4    0.920153
5    0.919519
Name: mae, dtype: float64
​
wqet_grader.grade("Project 3 Assessment", "Task 3.5.11", mae_series)
Yes! Your hard work is paying off.

Score: 1

Task 3.5.12: Look through the results in mae_series and determine what value for p provides the best performance. Then build and train final_model using the best hyperparameter value.

Note: Make sure that you build and train your model in one line of code, and that the data type of best_model is statsmodels.tsa.ar_model.AutoRegResultsWrapper.

mae_series.idxmax()
1
best_p = mae_series.idxmax()
best_model = AutoReg(y_train, lags=best_p).fit()
wqet_grader.grade(
    "Project 3 Assessment", "Task 3.5.12", [isinstance(best_model.model, AutoReg)]
)
That's the right answer. Keep it up!

Score: 1

Task 3.5.13: Calculate the training residuals for best_model and assign the result to y_train_resid. Note that your name of your Series should be "residuals".

y_train_resid = best_model.resid
y_train_resid.name = "residuals"
y_train_resid.head()
timestamp
2018-01-01 04:00:00+03:00    0.003846
2018-01-01 05:00:00+03:00   -0.013904
2018-01-01 06:00:00+03:00    0.247951
2018-01-01 07:00:00+03:00   -0.603135
2018-01-01 08:00:00+03:00   -0.509025
Freq: H, Name: residuals, dtype: float64
​
wqet_grader.grade("Project 3 Assessment", "Task 3.5.13", y_train_resid.tail(1500))
Yes! Your hard work is paying off.

Score: 1

Task 3.5.14: Create a histogram of y_train_resid. Be sure to label the x-axis as "Residuals" and the y-axis as "Frequency". Use the title "Best Model, Training Residuals".

# Plot histogram of residuals
y_train_resid.hist()
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title("Best Model, Training Residuals")
# Don't delete the code below 👇
plt.savefig("images/3-5-14.png", dpi=150)
​

with open("images/3-5-14.png", "rb") as file:
    wqet_grader.grade("Project 3 Assessment", "Task 3.5.14", file)
Good work!

Score: 1

Task 3.5.15: Create an ACF plot for y_train_resid. Be sure to label the x-axis as "Lag [hours]" and y-axis as "Correlation Coefficient". Use the title "Dar es Salaam, Training Residuals ACF".

fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")
plt.title("Dar es Salaam, Training Residuals ACF");
# Don't delete the code below 👇
plt.savefig("images/3-5-15.png", dpi=150)
​

with open("images/3-5-15.png", "rb") as file:
    wqet_grader.grade("Project 3 Assessment", "Task 3.5.15", file)
You got it. Dance party time! 🕺💃🕺💃

Score: 1

2.3. Evaluate
Task 3.5.16: Perform walk-forward validation for your model for the entire test set y_test. Store your model's predictions in the Series y_pred_wfv. Make sure the name of your Series is "prediction" and the name of your Series index is "timestamp".

y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = AutoReg(history, lags=best_p).fit()
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])
    
    
y_pred_wfv.name = "prediction"
y_pred_wfv.index.name = "timestamp"
y_pred_wfv.head()
/tmp/ipykernel_143/2767775534.py:1: DeprecationWarning: The default dtype for empty Series will be 'object' instead of 'float64' in a future version. Specify a dtype explicitly to silence this warning.
  y_pred_wfv = pd.Series()
timestamp
2018-03-06 00:00:00+03:00    8.136259
2018-03-06 01:00:00+03:00    9.338934
2018-03-06 02:00:00+03:00    7.323142
2018-03-06 03:00:00+03:00    6.588160
2018-03-06 04:00:00+03:00    7.396187
Freq: H, Name: prediction, dtype: float64
​
wqet_grader.grade("Project 3 Assessment", "Task 3.5.16", y_pred_wfv)
Yup. You got it.

Score: 1

Task 3.5.17: Submit your walk-forward validation predictions to the grader to see test mean absolute error for your model.

wqet_grader.grade("Project 3 Assessment", "Task 3.5.17", y_pred_wfv)
---------------------------------------------------------------------------
Exception                                 Traceback (most recent call last)
Input In [48], in <cell line: 1>()
----> 1 wqet_grader.grade("Project 3 Assessment", "Task 3.5.17", y_pred_wfv)

File /opt/conda/lib/python3.9/site-packages/wqet_grader/__init__.py:180, in grade(assessment_id, question_id, submission)
    175 def grade(assessment_id, question_id, submission):
    176   submission_object = {
    177     'type': 'simple',
    178     'argument': [submission]
    179   }
--> 180   return show_score(grade_submission(assessment_id, question_id, submission_object))

File /opt/conda/lib/python3.9/site-packages/wqet_grader/transport.py:143, in grade_submission(assessment_id, question_id, submission_object)
    141     raise Exception('Grader raised error: {}'.format(error['message']))
    142   else:
--> 143     raise Exception('Could not grade submission: {}'.format(error['message']))
    144 result = envelope['data']['result']
    146 # Used only in testing

Exception: Could not grade submission: Could not verify access to this assessment: Received error from WQET submission API: You have already passed this course!
3. Communicate Results
Task 3.5.18: Put the values for y_test and y_pred_wfv into the DataFrame df_pred_test (don't forget the index). Then plot df_pred_test using plotly express. Be sure to label the x-axis as "Date" and the y-axis as "PM2.5 Level". Use the title "Dar es Salaam, WFV Predictions".

df_pred_test = pd.DataFrame(
    {'y_test':y_test, 'y_pred_wfv':y_pred_wfv})
fig = px.line(df_pred_test, labels={'Value':'PM2.5'})
fig.update_layout(
    title="Dar es Salaam, WFV Predictions",
    xaxis_title="Date",
    yaxis_title="PM2.5 Level",
)
# Don't delete the code below 👇
fig.write_image("images/3-5-18.png", scale=1, height=500, width=700)
​
fig.show()
with open("images/3-5-18.png", "rb") as file:
    wqet_grader.grade("Project 3 Assessment", "Task 3.5.18", file)
---------------------------------------------------------------------------
Exception                                 Traceback (most recent call last)
Input In [52], in <cell line: 1>()
      1 with open("images/3-5-18.png", "rb") as file:
----> 2     wqet_grader.grade("Project 3 Assessment", "Task 3.5.18", file)

File /opt/conda/lib/python3.9/site-packages/wqet_grader/__init__.py:180, in grade(assessment_id, question_id, submission)
    175 def grade(assessment_id, question_id, submission):
    176   submission_object = {
    177     'type': 'simple',
    178     'argument': [submission]
    179   }
--> 180   return show_score(grade_submission(assessment_id, question_id, submission_object))

File /opt/conda/lib/python3.9/site-packages/wqet_grader/transport.py:143, in grade_submission(assessment_id, question_id, submission_object)
    141     raise Exception('Grader raised error: {}'.format(error['message']))
    142   else:
--> 143     raise Exception('Could not grade submission: {}'.format(error['message']))
    144 result = envelope['data']['result']
    146 # Used only in testing

Exception: Could not grade submission: Could not verify access to this assessment: Received error from WQET submission API: You have already passed this course!
