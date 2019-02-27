AI Challenger 2018 Weather Forecasting TOP3 Solution
==============================
> Give more accurate weather forecasts using single models based on machine learning.

It is lucky to have a top-5 rank because there is evident gap between us and the other top players, even some seriously great team who rank after us for forgetting to submit results every day or have a very bad results in the early days. So we just share our naive idea and welcome join us to have a discussion and play competitions together!
### Requirements
Codes are run on **Windows**. It is based on **Python 3.6.** Required packages are included in **requirements.txt**. Run bellow command to install them.
> pip install -r requirements.txt

### Pipeline for quick start.
All raw data are in ./data, along with data that we have processed into a single .csv file which is in all_data.zip, you can first unzip this file and then you can run those code.

Or you want to download by yourself, then you can go to https://challenger.ai/competition/wf2018. to downlowd 3 datasets as bellow (You can switch to English from top-right corner):

Training set: **ai_challenger_wf2018_trainingset_20150301-20180531.nc**  
Validation set: **ai_challenger_wf2018_validation_20180601-20180828_20180905.nc**  
Test set (Taking one-day test data on 28/10/2018 as an example): **ai_challenger_wf2018_testb1_20180829-20181028**

After downloaded, set three original dataset into the folder ./data and then process all data into one single csv
(For quick start, we just take **ai_challenger_wf2018_testb1_20180829-20181028** as a test example).Then we provide functions combine_data(file_names) in ./code/data_utils.py which is used to combine all data into one single .csv file.

(file_names = ['ai_challenger_wf2018_trainingset_20150301-20180531.nc','ai_challenger_wf2018_validation_20180601-20180828_20180905.nc', 'ai_challenger_wf2018_testb7_20180829-20181103.nc'])

### Train Models

We use different models in different days.

  1. We firstly try some baseline models using slide windows and predict the observation data straightly in 'train_windows_model.py' during Test A1.

  2. We then use old days feature and predict the difference between ruitu and real data in 'train_old_days_model.py' during Test A2 - Test B1 (A bad mistake is in Test A2's submit data to have a terrible result).

  3. After B1 we try more catlog features in 'train_old_days_model_catlog.py' during Test B2 - Test B4 because we find that **by predicting the difference between ruitu and real data** the results are getting better.

  4. We then add more features into the last model in 'train_old_days_model_catlog2.py' during Test B5 - Test B7.

  5. In the last two days other teammates also tried different features in './final_days/train_new.py' and './final_days/train_new2.py' during Test B6 - Test B7.

If you want to train those models, just **run the model python file in your terminal "python xxx.py"** (If the data is not ready, data process codes are included in these files.), except **'train_new.py/train_new2.py'** which you can **run 'python train_new.py/train_new2.py'** in your terminal (make sure that all_data.csv is in './code/final_days', sorry for bad organization).

PS: We use data before date '2018-10-15' because we find that data after that may cause a worse result.

### Test Models
Model testing codes are mainly included in 'test_models.py'.

1. For  **'windows_model', 'old_days_model', 'old_days_model_catlog', 'old_days_model_catlog2'**, you can **run the model python file in your terminal "python test_models.py --model model_name"**, in which 'model_name' is the model choosed to predict. The results of different days can be directly output when you assign 'obs_file' and 'anen_file' in 'test_models.py'. For example, when we assign:

	obs_file = "../data/ai_challenger_wf2018_testb5_obs_2018110103.csv"
	fore_file = "../data/ai_challenger_wf2018_testb5_fore_2018110103.csv"

 Then the result score will be results of Test B5's data.

2. For **'final_days'** model, **run "python train_new.py"** or **run "python train_new2.py",** the results every day will be computed automatically.

Hope this project will help you.


Best regards.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── code               <- All codes of this solution.
    │   └── final_days     <- The source code, model file and submitted file for another model in the last two days.
    ├── data               <- All data of this competition.
    │   └── tmp            <- Temporary Data.
    modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained models
    │   └── best_models    <- Best trained models(maybe).
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── output                <- The submitted file generated by test model file.

--------
