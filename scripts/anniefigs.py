import os,sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import*
#import pygmt
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Polygon
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from matplotlib.lines import Line2D
from glob import glob
from inspect import getfile, currentframe
from textwrap import wrap
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression

import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

from matplotlib import colors

# AX1
def ax1(eruption):

    # constants
    month = timedelta(days=365.25/12)

    # Create td object
    td = TremorData('WIZ')

    # Choose eruption to exclude from training _zsc
    te = td.tes[eruption]
    exclude_dates=[te-month,te+month]

    # All data logged
    allData=td.df
    # Log the rsam
    allData_logrsam=np.log10(allData['rsam'])

    # Histogram with markers on std and mean and a value
    f,ax= plt.subplots()
    plt.hist(allData_logrsam, bins='auto',color='lightblue') 
    plt.title("All data")
    ax.set_xlabel('log(RSAM)')
    # Get mean and std
    mean=allData_logrsam.mean()
    std=allData_logrsam.std()

    # Insert random value here to calculate zscore of
    # zsc=(2.654-mean)/std

    # Marking the mean and value
    plt.axvline(mean, color='k', linestyle='--', label='mean = {:f}'.format(mean))
    plt.annotate('x',xy=(mean+0.04,2050), xytext=(2.654, 2000), arrowprops=dict(arrowstyle="<->"),color='b')

    # Another way of plotting the arrow
    # plt.arrow(x=mean+0.05,y=2000,dx=2.649-mean-0.1,dy=0,length_includes_head = True,head_length=0.04,head_width=0.05)
    # plt.arrow(x=0.05+2.649-0.1,y=2000,dx=-(2.649-mean-0.1),dy=0,length_includes_head = True,head_length=0.04,head_width=0.05)
    # plt.plot([2.654], [2000], marker='x', markersize=6, color="blue")
    # plt.axvline(2.654, color='b', linestyle='--', label='zscore=1')

    plt.text(2.25, 2400,'z score=1',fontsize=10)
    plt.text(2.6, 1700,'2.654', color='blue',fontsize=8)
    ax.set_ylabel('frequency')
    plt.legend()
    plt.savefig('plots/FigA1.png')

    # Training data only (excluding one eruption) 
    t0=exclude_dates[0]
    t1=exclude_dates[1]
    inds = (allData.index<t0)|(allData.index>=t1)
    trainData = allData.loc[inds]
    # Log the rsam column
    trainData_logrsam=np.log10(trainData['rsam'])

    # Histogram with markers on std and mean and the same value  
    fig,ax2=plt.subplots()
    plt.hist(trainData_logrsam, bins='auto',color='lightblue') 
    ax2.set_xlabel('log(RSAM)')
    plt.axvline(mean, color='k', linestyle='--', label='mean = {:f}'.format(mean))
    # Calculate mean and standard deviation
    mean=trainData_logrsam.mean()
    std=trainData_logrsam.std()
    # Arrow to show z score
    plt.annotate('x',xy=(mean+0.04,2050), xytext=(2.637, 2000), arrowprops=dict(arrowstyle="<->"),color='b')

    # Alternate way
    # plt.plot([2.637], [2000], marker='x', markersize=6, color="blue")
    # plt.arrow(x=mean+0.05,y=2000,dx=2.632-mean-0.1,dy=0,length_includes_head = True)
    # plt.arrow(x=0.05+2.632-0.1,y=2000,dx=-(2.632-mean-0.1),dy=0,length_includes_head = True,head_width=0.1)
    # plt.axvline(2.637, color='b', linestyle='--', label='zscore=1')

    plt.title("Training data")
    plt.text(2.25,2400,'z score=1',fontsize=10)
    plt.text(2.57,1700,' 2.637', color='blue', fontsize=8)
    ax2.set_ylabel('frequency')
    plt.legend()
    plt.savefig('plots/FigA2.png')

# AX2
def ax2():

    # Get z score accuracy data
    data_zsc  = np.genfromtxt('forplotting/accuracy_zsc.txt',dtype=float,delimiter=',',skip_header=1)
    threshold = data_zsc[:,0]
    FP_zsc = data_zsc[:,1]
    TP_zsc = data_zsc[:,3]
    alert_fraction_zsc =data_zsc[:,5]*100
    # AP_zsc = FP_zsc+TP_zsc
    N_zsc=TP_zsc+FP_zsc-1
    # false_alert_rate_zsc = (AP_zsc-4)/AP_zsc
    # rate_zsc= 4/(FP_zsc + 4)
    # Calculate false alert rate
    rate_zsc=(TP_zsc-1)/N_zsc

    # Get raw accuracy data
    data  = np.genfromtxt('forplotting/accuracy_raw.txt',dtype=float,delimiter=',',skip_header=1)
    FP = data[:,1]
    TP = data[:,3]
    alert_fraction =data[:,5]*100
    # AP= FP+TP
    # false_alert_rate= (AP-4)/AP
    N=TP+FP-1
    # rate=4/(FP + 4)
    # Calculate the false alert rate
    rate=(TP-1)/N

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot
    ax.plot(threshold, alert_fraction, color='k', label='Alert duration - Raw')
    ax.plot(threshold, alert_fraction_zsc, color='k', linestyle='dashed', label='Alert duration - Zscore')  
    # set x-axis label
    ax.set_xlabel("Threshold")
    # set y-axis label
    ax.set_ylabel("Alert duration (%)")
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(threshold, rate, color="blue", label='Eruption probability during alert - Raw')
    ax2.plot(threshold, rate_zsc, color="blue", linestyle='dashed', label='Eruption probability during alert - Zscore')
    ax2.set_ylabel("Eruption probability during alert")
    ax.plot([],[], color="blue", label='Eruption probability during alert - Raw')
    ax.plot([],[], color="blue", linestyle='dashed', label='Eruption probability during alert - Zscore')
    ax.legend(loc='upper left')
    ax.set_xlim([0.6,1])
    plt.savefig("plots/FigC.png")

# AX3

def ax3(eruption):

    dates=['aug2012','aug2013','oct2013','apr2016','dec2019']

    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    td = TremorData('WIZ')

    # Choose eruption to exclude from training _zsc
    te = td.tes[eruption]
    # Untransformed fm object
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', station='WIZ', window=2., overlap=0.75, 
        look_forward=2., exclude_dates=[[te-month,te+month],], data_streams=['rsam','mf','hf'], root='{:s}_nodsarforecaster'.format(dates[eruption]))
    # Z score transformed object
    fm2 = ForecastModel(ti='2011-01-01', tf='2020-01-01', station='WIZ', window=2., overlap=0.75, 
        look_forward=2., exclude_dates=[[te-month,te+month],], data_streams=['zsc_rsam','zsc_mf','zsc_hf'], root='{:s}_10nodsarforecaster'.format(dates[eruption]))

    # columns to manually drop from feature matrix because they are highly correlated to other 
    # linear regressors
    drop_features = ['linear_trend_timewise','agg_linear_trend']

    # set the available CPUs higher or lower as appropriate
    n_jobs = 3

    # train the model, excluding 2019 eruption
    # note: building the feature matrix may take several hours, but only has to be done once 
    # and will intermittantly save progress in ../features/
    # trained scikit-learn models will be saved to ../models/*root*/
    fm.train(ti='2011-01-01', tf='2020-01-01',  Ncl=500, drop_features=drop_features, retrain=False, 
        exclude_dates=[[te-month,te+month],], n_jobs=n_jobs)      
    fm2.train(ti='2011-01-01', tf='2020-01-01',  Ncl=500, drop_features=drop_features, retrain=False, 
        exclude_dates=[[te-month,te+month],], n_jobs=n_jobs)      

    # # predict on hires features
    ys = fm.hires_forecast(ti=te-month/3, tf=te+month/10, recalculate=False, n_jobs=n_jobs)
    ys2 =fm2.hires_forecast(ti=te-month/3, tf=te+month/10, recalculate=False, n_jobs=n_jobs)
    
    # set up figure 
    f = plt.figure(figsize=(30,30))
    ax3 = plt.axes([0.1, 0.08, 0.8, 0.4])

    # Get _zsc range
    t = pd.to_datetime(ys.index.values)

    # Get ensemble mean for both models
    y = np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
    y2= np.mean(np.array([ys2[col] for col in ys2.columns]), axis=0)

    # axis labels
    # Get rsam
    rsam = fm.data.get_data(t[0], t[-1])['rsam']
    trsam = rsam.index
    ts = [t[-1], trsam[-1]]
    tmax = np.max(ts)
    ax3.set_xlim([tmax-timedelta(days=13), tmax])
    ax3.set_title(dates[eruption])
    ax3.set_ylim([-0.05, 1.05])
    ax3.set_yticks([0,0.25,0.50,0.75,1.00])
    ax3.set_ylabel('ensemble mean')

    # consensus threshold
    ax3.axhline(0.8, color='k', linestyle=':', label='alert threshold', zorder=10)

    # Plotting the models against each other
    ax3.plot(t, y, 'c-', label='ensemble mean raw', color='green', zorder=6, lw=0.75)
    ax3.plot(t,y2,'c-', label='ensemble mean z score', color='blue', zorder=4, lw=0.75)

    # Plot rsam and labels
    ax_ = ax3.twinx()
    ax_.set_ylabel('RSAM [$\mu$m s$^{-1}$]')
    ax_.set_ylim([0,5])
    ax_.set_xlim(ax3.get_xlim())
    ax_.plot(trsam, rsam.values*1.e-3, 'k-', lw=0.75)

    # Fill in forecast period
    for tii,yi in zip(t, y):
        if yi > 0.8:
            ax3.fill_between([tii, tii+fm.dtf], [0,0], [0.48,0.48], color='y', zorder=2)

    for tii,yi in zip(t, y2):
        if yi > 0.8:
            ax3.fill_between([tii, tii+fm.dtf], [0.52,0.52], [1,1], color='cyan', zorder=2)


    ax3.fill_between([], [], [], color='y', label='eruption forecast raw')
    ax3.fill_between([], [], [], color='blue', label='eruption forecast z score')

    ax3.plot([],[],'k-', lw=0.75, label='RSAM')
    
    # Set ticks
    tf = tmax 
    t0 = tf.replace(hour=0, minute=0, second=0)
    xts = [t0 - timedelta(days=i) for i in range(13)][::-1]
    lxts = [xt.strftime('%d %b') for xt in xts]
    ax3.set_xticks(xts)
    ax3.set_xticklabels(lxts,rotation=45)
    ax3.set_xlabel('UTC')

    # Last axes for zscore value
    ax_zsc = ax3.twinx()
    zsc = fm2.data.get_data(t[0], t[-1])['zsc_rsam']
    tzsc = zsc.index

    # Floating axis for z score
    # Offset the right spine of ax_zsc
    rspine = ax_zsc.spines['right']
    rspine.set_position(('axes', 1.05))
    ax_zsc.set_frame_on(True)
    ax_zsc.plot(tzsc,zsc,color='grey',linestyle='dashed', zorder=8, lw=0.75, label='z score')
    ax_zsc.set_ylim([0,750])
    # plt.locator_params(axis='y', nbins=10)
    ax_zsc.set_ylabel("Z score")
    ax3.plot([],[],color='grey',linestyle='dashed', lw=0.75, label='z score')

    # eruption and legend
    ax3.axvline(te, color='r', linestyle='dashed', label='eruption',zorder=12)  
    ax3.legend(loc='upper left', ncol=3)
    
    plt.savefig('plots/{:s}Fig.png'.format(dates[eruption]))
    plt.show()

if __name__ == '__main__':
    ax1(0)
    # ax2()
    # ax3(0)
    # ax3(1)
    # ax3(2)
    # ax3(3)
    # ax3(4)