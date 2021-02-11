import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, load_dataframe
from datetime import timedelta, datetime
import numpy as np

# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)


def forecast_dec2019():
    ''' forecast model for Dec 2019 eruption
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    td = TremorData('WIZ')
        
    # construct model object
    # 'mf','hf','dsar'
    data_streams = ['zsc_rsam']
    # data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', station='WIZ', window=2., overlap=0.75, 
        look_forward=2., data_streams=data_streams, root='dec2019_zscforecaster')
    # fm.data.plot()
    
    # columns to manually drop from feature matrix because they are highly correlated to other 
    # linear regressors
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 3

    # train the model, excluding 2019 eruption
    # note: building the feature matrix may take several hours, but only has to be done once 
    # and will intermittantly save progress in ../features/
    # trained scikit-learn models will be saved to ../models/*root*/
    te = td.tes[-1]
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, retrain=False, 
        exclude_dates=[[te-month,te+month],], n_jobs=n_jobs)      

    # run forecast from 2011 to 2020
    # model predictions will be saved to ../predictions/*root*/ 
    ys = fm.forecast(ti='2011-01-01', tf='2020-01-01', recalculate=True, n_jobs=n_jobs)    

    # plot forecast and quality metrics
    # plots will be saved to ../plots/*root*/
    fm.plot_forecast(ys, year= 2011 ,threshold=0.8, xlim = [te-month/4., te+month/15.], 
        save=r'{:s}/forecast.png'.format(fm.plotdir))
    fm.plot_accuracy(ys, save=r'{:s}/accuracy.png'.format(fm.plotdir))

    # construct a high resolution forecast (10 min updates) around the Dec 2019 eruption
    # note: building the feature matrix might take a while
    fm.hires_forecast(ti=te-month/3, tf=te+month/8, recalculate=True, 
        save=r'{:s}/dec2019_forecast_hires.png'.format(fm.plotdir), n_jobs=n_jobs)

def forecast_WIZ():
    ''' forecast model for Dec 2019 eruption
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    td = TremorData('WIZ')
        
    # construct model object
    # data_streams = ['pct_rsam']
    # data_streams = ['rsam','mf','hf','dsar']
    # data_streams = ['zsc_rsam','zsc_mf','zsc_hf']
    data_streams = ['rsam','mf','hf']

    te = td.tes[0]
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', station='WIZ', window=2., overlap=0.75, 
        look_forward=2., exclude_dates=[[te-month,te+month],], data_streams=data_streams, root='aug2012_nodsarforecaster')
    
    # columns to manually drop from feature matrix because they are highly correlated to other 
    # linear regressors
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 3

    # train the model, excluding 2019 eruption
    # note: building the feature matrix may take several hours, but only has to be done once 
    # and will intermittantly save progress in ../features/
    # trained scikit-learn models will be saved to ../models/*root*/
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, retrain=False, 
        exclude_dates=[[te-month,te+month],], n_jobs=n_jobs)      
    

    # run forecast from 2011 to 2020
    # model predictions will be saved to ../predictions/*root*/ 
    ys = fm.forecast(ti='2011-01-01', tf='2020-01-01', recalculate=False, n_jobs=n_jobs)    

    # plot forecast and quality metrics
    # plots will be saved to ../plots/*root*/
    # fm.plot_forecast(ys, year= 2011 ,threshold=0.8, xlim = [te-month/4., te+month/15.], 
    #     save=r'{:s}/forecast.png'.format(fm.plotdir))
    # fm.plot_accuracy(ys, save=r'{:s}/accuracy.png'.format(fm.plotdir))

    # construct a high resolution forecast (10 min updates) around the Dec 2019 eruption
    # note: building the feature matrix might take a while
    # fm.hires_forecast(ti=te-month/3, tf=te+month/8, recalculate=True, 
    #     save=r'{:s}/dec2019_forecast_hires.png'.format(fm.plotdir), n_jobs=n_jobs)

def forecast_test():
    ''' test scale forecast model
    '''
    # constants
    month = timedelta(days=365.25/12)
        
    # set up model
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2012-07-15', tf='2012-08-16', window=2., overlap=0.75, 
        look_forward=2., data_streams=data_streams, root='test', savefile_type='pkl')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 1
    
    # train the model
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2012-07-15', tf='2012-08-16', drop_features=drop_features, retrain=False,
        n_jobs=n_jobs)      

    # plot a forecast for a future eruption
    # tf = te+month/30
    # fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=True, 
    #     save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

    te = fm.data.tes[1]
    y = load_dataframe(r'D:\code\whakaari\predictions\test_hires\DecisionTreeClassifier_0000.pkl')
    tf = y.index[-1] + month/30./10.
    fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=tf, recalculate=False, 
        save=r'{:s}/forecast_Aug2013.png'.format(fm.plotdir), n_jobs=n_jobs)

def forecast_now():
    ''' forecast model for present day 
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
        
    # pull the latest data from GeoNet
    td = TremorData('WIZ')
    # td.update()

    # model from 2011 to present day (td.tf)
    # data_streams = ['rsam','mf','hf','dsar']
    data_streams = ['zsc_rsam','zsc_mf','zsc_hf']
    # data_streams = ['rsam','mf','hf']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', station='WIZ', window=2, overlap=0.75,  
        look_forward=2, data_streams=data_streams, root='10zsc_nodsar_forecaster')

    # f,ys=fm.get_features()
    # print(f.isnull().sum().sum())
    # print(ys.isnull().sum().sum())
    
    # set the available CPUs higher or lower as appropriate
    n_jobs =3
    
    # The online forecaster is trained using all eruptions in the dataset. It only
    # needs to be trained once, or again after a new eruption.
    # (Hint: feature matrices can be copied from other models to avoid long recalculations
    # providing they have the same window length and data streams. Copy and rename 
    # to *root*_features.csv)
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2011-01-01', tf='2020-01-01', Ncl=500, drop_features=drop_features, 
        retrain=True, n_jobs=n_jobs)      
    
    # forecast the last 7 days at high resolution
    # fm.hires_forecast(ti=fm.data.tf - 7*day, tf=fm.data.tf, recalculate=True, 
        # save='current_forecast.png', nztimezone=True, n_jobs=n_jobs)  
    

    # lores forecast
    # run forecast from 2011 to 2020
    # model predictions will be saved to ../predictions/*root*/ 
    ys = fm.forecast(ti='2011-01-01', tf='2020-01-01', recalculate=True, n_jobs=n_jobs)    

    # # plot forecast and quality metrics
    # # plots will be saved to ../plots/*root*/
    fm.plot_forecast(ys, year=2011, threshold=0.8, 
        save=r'{:s}/forecast.png'.format(fm.plotdir))
    fm.plot_accuracy(ys, save=r'{:s}/accuracy.png'.format(fm.plotdir))

    FP, FN, TP, TN, dur, mcc = fm._model_alerts(ys.index, ys['consensus'], 0.8, fm.look_forward/((1-fm.overlap)*fm.window),  timedelta(days=(1-fm.overlap)*fm.window))
    print("The false alarm rate is: ",(FP-4)/FP)

def forecast_AVO():
    ''' 
    forecast model for alaksan volcano
    '''
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)

    # train with whakaari data
        
    # pull the latest data from GeoNet
    td1 = TremorData('WIZ')
    # td1.update()

    # model from 2011 to present day (td.tf)
    data_streams = ['rsam','mf','hf','dsar']
    fm1 = ForecastModel(ti='2011-01-01', tf=td1.tf, station='WIZ', window=2, overlap=0.75,  
        look_forward=2, data_streams=data_streams, root='whakaari_forecaster')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs =3
    
    # The online forecaster is trained using all eruptions in the dataset. It only
    # needs to be trained once, or again after a new eruption.
    # (Hint: feature matrices can be copied from other models to avoid long recalculations
    # providing they have the same window length and data streams. Copy and rename 
    # to *root*_features.csv)
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm1.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, 
        retrain=False, n_jobs=n_jobs)      
    
    # run forecast from 2011 to 2020
    # model predictions will be saved to ../predictions/*root*/ 
    ys1 = fm1.forecast(ti='2011-01-01', tf='2020-01-01', recalculate=False, n_jobs=n_jobs)    

    # # plot forecast and quality metrics
    # # plots will be saved to ../plots/*root*/
    # fm1.plot_forecast(ys1, threshold=0.8, xlim = [te1-month/4., te1+month/15.], 
    #     save=r'{:s}/forecast.png'.format(fm1.plotdir))
    # fm1.plot_accuracy(ys1, save=r'{:s}/accuracy.png'.format(fm1.plotdir))

    # test with Pavlof
    # Create tremor data object
    td2 = TremorData('SSBA')
    # td2.df = td2.df['2006-01-01 00:00:00':]
    # td2.update()

    # model from 2005 to present day (td.tf)
    fm2 = ForecastModel(ti='2009-01-03', tf='2021-01-01', station='SSBA', savefile_type='pkl', window=2, overlap=0.75,  
        look_forward=2, data_streams=data_streams, root='shishaldin_forecaster')
        
    # standardize
    # Take log
    # WKR = np.log10(td1.df)
    # fm2.data.df=np.log10(fm2.data.df)

    # mean and standard deviation shift
    for freq in ['rsam','mf','hf','dsar']:
        # fm2.data.df[freq]=(fm2.data.df[freq]-fm2.data.df[freq].mean())/fm2.data.df[freq].std()*WKR[freq].std()+WKR[freq].mean()
        # # Find percentile of value
        fm2.data.df[freq]=np.percentile(fm1.data.df[freq],fm2.data.df[freq].rank(pct=True)*100)
    
    # convert back from log
    # fm2.data.df=10**fm2.data.df
    
    # run forecast from 2006 to 2020
    # model predictions will be saved to ../predictions/*root*/ 
    ys2 = fm2.forecast(ti='2009-01-03', tf='2021-01-01', use_model=fm1.modeldir, recalculate=True, n_jobs=n_jobs)    

    # # plot forecast and quality metrics
    # # plots will be saved to ../plots/*root*/
    fm2.plot_forecast(ys2, year=2009, threshold=0.8, xlim = [td2.tes[-1]-month/4., td2.tes[-1]+month/15.], 
        save=r'{:s}/forecast.png'.format(fm2.plotdir))
    fm2.plot_accuracy(ys2, save=r'{:s}/accuracy.png'.format(fm2.plotdir))

    # construct a high resolution forecast (10 min updates) around eruptions
    # note: building the feature matrix might take a while
    # -fm2.dtw-fm2.dtf-day*6
    for te in td2.tes:
        fm2.hires_forecast(ti=te-month/3, tf=te+month/8, root='SSBA_hires_{:d}-{:d}-{:d}'.format(te.day,te.month,te.year),recalculate=True, 
            save=r'{:s}/forecast_hires_-{:d}-{:d}-{:d}.png'.format(fm2.plotdir,te.day,te.month,te.year), n_jobs=n_jobs,use_model=fm1.modeldir)

if __name__ == "__main__":
    # forecast_dec2019()
    # forecast_test()
    forecast_WIZ()
    # forecast_now()
    # forecast_AVO()
