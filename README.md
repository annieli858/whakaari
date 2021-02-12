# Changes made by Annie

1. Line 87 
- Made station a non-default input
- Added exclude dates attribute for zscore transformation

2. Line 91
- Defining the exclude_dates

3. Line 103 
- Added formatting to open eruptive periods of any given station

4. Line 141-181
- Percentile and z score transformations

5. Line 491
- Removed station = 'WIZ' (so its non-default)
- added exclude_dates as empty array by default

6. Line 497&498
- Moved up exclude_dates initialization so its done before the transform is done (creating the TremorData object) This replaces the initialization in line 530.
- Passed in exclude_dates into creating TremorData object

7. Lines 599&600
- Prevents non-negativity error message

8. Line 1010
-  Added use_model=None (However this was more for testing on the Alaskan Volcano stuff)

9. Line 1094-1098
- Added condition for use_model to avoid the "must train model before testing" error ((However this was more for testing on the Alaskan Volcano stuff)

10. Line 1240/1267
- Added use_model=None to bypass 'Train model before constructing hires forecast.' error (For Alaskan stuff)

11. Line 1282
- Added self.exclude_dates as input (For z score transformation)

12. Line 1287
- Another condition for use_model

13. Line 1293
- The te input depends on the ti input into the hires_forecast function. (eg. ti=te-month/3) This ultimately reverses the process of starting the hires forecast xx days before the eruption to plot the eruption line. (eg. if ti=te-month/3, te=ti+month/3) 

14. Line 1297
- Added a new input year- this is when you want the lores forecast to start plotting (I did this because it was hard coded to 2011 before)

15. Line 1318, 1325-1331, 1367-1364
- Replacing the hardcoded times with 'year'

16. Line 1351
- Commented this out as it was raising errors for when the training data was not the same as testing data (For Alaskan volcano stuff)

17. Line 1386
- Added the te input from before (Change No.13)

18. Lines 1421- 1428/ Line 1464
- Changed the hires plotting window from 7 to 15
- Set the title to depend on the root
- Plotted a red dashed line for the eruption time based on the new input te

19. Lines 1474-1477
- Tried to make this not hard coded but I think this is for the upper half of the hires forecast- so this doesn't matter as much

20. Line 1804
- Added IRIS network codes for Alaskan stuff

21. Line 1894
- AJ's change from (ti+j*600) to (ti+(j+1)*600)

22. Line 1902
- Added input 'WIZ' to be consistent with change in TremorData object

23. Line 1933
- Added random seed (Talked about this in meeting)

###### ###### ###### ###### ###### ###### ###### ###### ###### ######

# Whakaari
Eruption forecast model for Whakaari (White Island, New Zealand). This model implements a time series feature engineering and classification workflow that issues eruption alerts based on real-time tremor data.

## Installation

Ensure you have Anaconda Python 3.7 installed. Then

1. Clone the repo

```bash
git clone https://github.com/ddempsey/whakaari
```

2. CD into the repo and create a conda environment

```bash
cd whakaari

conda env create -f environment.yml

conda activate whakaari_env
```

The installation has been tested on Windows, Mac and Unix operating systems. Total install with Anaconda Python should be less than 10 minutes.

## Running models
Three examples have been included in ```scripts/forecast_model.py```. 

The first, ```forecast_test()``` trains on a small subset of tremor data in 2012 and then constructs a forecast of the Aug 2013 eruption. It will take about 10 minutes to run on a desktop computer and produces a forecast image in ../plots/test/forecast_Aug2013.png

The second, ```forecast_dec2019()``` trains on tremor data between 2011 and 2020 but *excluding* a two month period either side of the Dec 2019 eruption. It then constructs a model 'forecast' of this event. It could take several hours or a day to run depending on the cpus available for your computer.

The third, ```forecast_now()```, is an online forecaster. It trains a model on all data between 2011 and 2020 *including* the Dec 2019 eruption. It then downloads the latest Whakaari tremor data from GeoNet and constructs a forecast for the next 48 hours. See associated paper for guidance on interpreting model consensus levels. The model may take several hours or a day the first time it is constructed, but subsequent updates should be quick.

To run the models, open ```forecast_model.py```, comment/uncomment the forecasts you want to run, then in a terminal type
```bash
cd scripts

python forecast_model.py
```

## Disclaimers
1. This eruption forecast model is not guaranteed to predict every future eruption, it only increases the likelihood. In our paper, we discuss the conditions under which the forecast model is likely to perform poorly.

2. This eruption forecast model provides a probabilistic prediction of the future. During periods of higher risk, it issues an alert that an eruption is *more likely* to occur in the immediate future. At present, our best estimate is that when an alert is issued at 80% consensus there is a 1 in 12 chance of an eruption occurring. On average, alerts last about 5 days. Eruption probability and average alert length could change depending on which eruptions are used to train the model.

3. This software is not guaranteed to be free of bugs or errors. Most codes have a few errors and, provided these are minor, they will have only a marginal effect on accuracy and performance. That being said, if you discover a bug or error, please report this at [https://github.com/ddempsey/whakaari/issues](https://github.com/ddempsey/whakaari/issues).

4. This is not intended as an API for designing your own eruption forecast model. Nor is it designed to be especially user-friendly for Python/Machine Learning novices. Nevertheless, if you do want to adapt this model for another volcano, we encourage you to do that and are happy to answer queries about the best way forward. 

## Acknowledgments
This eruption model would not be possible without real-time seismic data streaming from Whakaari via GeoNet.

