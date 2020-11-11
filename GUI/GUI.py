import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation
import os
from datetime import datetime 
import subprocess
import pygame
import random
from pyOpenBCI import OpenBCICyton
from pylsl import StreamInfo, StreamOutlet
from work_session import EEGSampler
from EEGRecorder import EEGRecorder
from CSVWriter import CSVWriter
import time
import numpy as np
import pickle
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
import pyeeg
from data_processing_helper import * 
import warnings
warnings.filterwarnings('ignore')
# Import time-frequency functions
from neurodsp.timefrequency import amp_by_time, freq_by_time, phase_by_time
from neurodsp.plts.time_series import plot_time_series, plot_instantaneous_measure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import accuracy_score




# Trials
# A ton of constants
num_trials = 1  #change this for testing purposes
a1 = [1]*num_trials
a2 = [2]*num_trials
# a3 = [3]*num_trials
# a4 = [4]*num_trials
a_123 = a1 + a2

# shuffle
random.shuffle(a_123)

# Fonts
LARGE_FONT= ("Verdana", 12)
BOLD_FONT= ("Verdana", 12, 'bold')
UNDERLINE_FONT = ("Verdana", 12, 'underline')

# Filenames
BEEP_FILENAME = './audio/a.mp3'
EEG_OUTPUT_FILENAME = "./data/eeg_data.csv"
EVENT_OUTPUT_FILENAME = "./data/event_data.csv"
Pkl_Filename = "PowerBinModel.pkl"

# Trial timing
START_REST_TIME = 1000 
BEEP_FINISH_TIME = START_REST_TIME + 500 # 0.5s to play beep
ARROW_FINISH_TIME = BEEP_FINISH_TIME + 1000 # 1s to show arrow
TRIAL_FINISH_TIME = ARROW_FINISH_TIME + 4000 # 4s for actual trial
TRIAL_BASEEXTRA_TIME = TRIAL_FINISH_TIME + 500 # 0.5s to finish trial
MAX_EXTRA_TIME = 1000 # randomly give an extra 0 to MAX_EXTRA_TRIAL time between trials


def getPowerRatio(eeg_data, binning, eeg_fs=250):
    power, power_ratio = pyeeg.bin_power(eeg_data, binning, eeg_fs)
    return np.array(power_ratio)
def getIntervals(binning): 
    intervals = list()
    for i, val in enumerate(binning[:-1]): 
        intervals.append((val, binning[i+1]))
    return intervals
def getOutputLabelsAndEpochTimes(event_df):
    # Generates the ordered list of output labels and epoch time pairs
    # Input: event_df -- the event dataframe from the csv file
    # Output: 
    #    output_labels -- [1, 2, 4, 3, etc] where the integers correspond to the trial type encoded
    #    epoch_times -- [[<timestamp of start>, <timestamp of end>], [<timestamp of start>, <timestamp of end>], etc]
    
    output_labels = []
    epoch_times = []
    current_epoch = []
    for index, row in event_df.iterrows():
        event_info = row['EventStart'].split("_")
        if event_info[0] == 'start': 
            output_labels.append(int(event_info[1]))
            current_epoch.append(row['time'])
        else :
            current_epoch.append(row['time'])
            epoch_times.append(list(current_epoch))
            current_epoch = []
    return np.array(output_labels), np.array(epoch_times)
def getEEGEpochs(epoch_times, eeg_df, target_num_trials=1000):
    # Slices and generates the epochs in the eeg_df given the epoch_times
    # Input: 
    #    epoch_times: [[<timestamp of start>, <timestamp of end>], [<timestamp of start>, <timestamp of end>], etc]
    #    eeg_df: dataframe from csv file
    # Output: 
    #    a numpy array containing eeg_epochs (#epoch, #chans, #timepoints)
    eeg_epochs = []
    for epoch_time in epoch_times: 
        baseline_df = eeg_df[(eeg_df['time'] > (epoch_time[0] - 1.5)) & (eeg_df['time'] < ((epoch_time[0])))]
        baseline_df = baseline_df.drop(columns=['time'])
        baselines = np.mean(baseline_df[eeg_chans].values, 0)
        sub_df = eeg_df[(eeg_df['time'] > epoch_time[0]) & (eeg_df['time'] < epoch_time[1])]
        sub_df = sub_df.drop(columns=['time'])
        sub_df = sub_df - baselines
        num_above = len(sub_df) - target_num_trials
        if num_above >= 0:
            epoch = np.array(sub_df.values[num_above // 2: len(sub_df) - num_above // 2])[:1000]
            eeg_epochs.append(epoch.T)
            if len(epoch) != 1000:
                print("Warning: Potential off by 1 error. Found trail with != 1000 samples:", len(epoch))
        else: 
            print("Warning: Epoch with less than", target_num_trials, "eeg samples")
    return np.array(eeg_epochs)


# Model Definitions##############################################################################
# Even though the model is loaded from a file, its functionality still needs to be defined again
class myModel:
    def __init__(self):
        pass
    def fit(X, Y):
        pass
    def predict(X):
        pass
    
class XDawnLRModel(myModel): # XDAWN Covariance Preprocessing + Linear Regression Classifier
    def __init__(self):
        super().__init__()
        self.XC = XdawnCovariances(nfilter = 1) # the number of filters can be changed
        self.logreg = LogisticRegression()
        
    def fit(self, X, Y):
        X_transformed = self.XC.fit_transform(X, Y)
        X_transformed = TangentSpace(metric='riemann').fit_transform(X_transformed)
        return self.logreg.fit(X_transformed,Y)
        
    def predict(self, X):
        X_transformed = self.XC.transform(X)
        X_transformed = TangentSpace(metric='riemann').fit_transform(X_transformed)
        return self.logreg.predict(X_transformed)
    
    def predict_proba(self, X):
        X_transformed = self.XC.transform(X)
        X_transformed = TangentSpace(metric='riemann').fit_transform(X_transformed)
        return self.logreg.predict_proba(X_transformed)
    
# Power Bin Model (may have other decision algorithms)
class PowerBinModel(myModel): # XDAWN Covariance Preprocessing + Linear Regression Classifier
    def __init__(self, chans, num_top=5):
        super().__init__()
        self.mod_types = [SVC, LogisticRegression, LinearDiscriminantAnalysis]
        self.mod_type = self.mod_types[0] 
        self.binning = [1, 3, 5, 7, 9, 12, 20, 40] #[0.5, 4, 7, 12, 30]
        self.intervals = getIntervals(self.binning)
        self.model = self.mod_type()
        self.scaler = StandardScaler()
        self.chans = chans
        self.feat_names = [str(ch) + "_" + str(ints) for ch in self.chans for ints in self.intervals]
        self.feat_names.extend([str(ch) + "_mobility" for ch in self.chans])
        self.feat_names.extend([str(ch) + "_complexity" for ch in self.chans])
        self.feat_names.extend(['left_'+ str(interval) for interval in self.intervals])
        self.feat_names.extend(['right_'+ str(interval) for interval in self.intervals])
        self.feat_names.extend(['left_minus_right_'+ str(interval) for interval in self.intervals])

        self.feature_indx = None
        self.num_top = num_top
        
    def getFeatureNames(self):
        return self.feat_names
        
    def _getFeatures(self, eeg_datas) :
        feats = []
        mobs = []
        comps = []
        left_powers = {}
        right_powers = {}
        for i, eeg_data in enumerate(eeg_datas): 
            ratios = getPowerRatio(eeg_data, self.binning)
            
            feats.extend(ratios)

            mob, comp = pyeeg.hjorth(eeg_data)
            mobs.append(mob)
            comps.append(comp)
            ch = self.chans[i]
            for i, interval in enumerate(self.intervals) : 
                if (interval not in left_powers) or (interval not in right_powers): 
                    left_powers[interval] = 0
                    right_powers[interval] = 0
                if int(ch[1]) % 2 == 0: 
                    right_powers[interval] += ratios[i]
                else : 
                    left_powers[interval] += ratios[i]
        feats.extend(mobs)
        feats.extend(comps)
        left_powers_list = [] 
        right_powers_list = []
        for i, interval in enumerate(self.intervals) : 
            left_powers_list.append(left_powers[interval])
            right_powers_list.append(right_powers[interval])
        feats.extend(left_powers_list)
        feats.extend(right_powers_list)
        feats.extend(np.array(left_powers_list) - np.array(right_powers_list))
        
        return feats
    
    def _selectTopFeatures(self, X_features, Y) :
        # Get the unique classes of Y to be able to separate the features 
        unique_Y = np.unique(Y)
        X_features_mean = []
        X_features_std = []
        for un in unique_Y: 
            X_features_mean.append(np.mean(X_features[Y == un, :], 0))
            X_features_std.append(np.std(X_features[Y == un, :], 0) / np.sqrt(len(X_features[Y == un])))
        X_features_std_sum = np.sum(X_features_std, 0)
        X_features_mean_diff = np.abs(X_features_mean[0] - X_features_mean[1])
        X_features_mean_diff = X_features_mean_diff - X_features_std_sum
        
        # Get the index of the sorted mean difference array
        args = np.argsort(X_features_mean_diff)
        # Reverse args to largest features are first
        args = args[::-1]
        feature_indx = args[:self.num_top]
        X_features = np.transpose([X_features[:, i] for i in feature_indx])
        return X_features, feature_indx

    def fit(self, X, Y):
        # X shape must be (#Trials, #Chans, #Timepoints)
        
        # Check which model is the best 
        max_accs, best_model_idx = self.evaluate(X, Y)
        print("max accuracy:", max_accs)
        print("best model index:", best_model_idx)
        self.mod_type = self.mod_types[best_model_idx]
        self.model = self.mod_type()
        
        X_features = np.array([self._getFeatures(eeg_datas) for eeg_datas in X])
        X_features, self.feature_indx = self._selectTopFeatures(X_features, Y)
        
        X_features = self.scaler.fit_transform(X_features)
        self.model.fit(X_features,Y)
    
    def evaluate(self, X, Y): 
        unique_Y = np.unique(Y)
        loo = LeaveOneOut()
        X_features = np.array([self._getFeatures(eeg_datas) for eeg_datas in X])
        accs = [] 
        for mod in self.mod_types: 
            y_pred = []
            y_true = []
            for train_ix, test_ix in loo.split(Y):
                # split data
                X_train_i, X_test_i = X_features[train_ix, :], X_features[test_ix, :]
                y_train_i, y_test_i = Y[train_ix], Y[test_ix]

                X_train_i, feature_indx = self._selectTopFeatures(X_train_i, y_train_i)
                scaler = StandardScaler()
                X_train_i = scaler.fit_transform(X_train_i)

                # fit model
                model = mod()
                model.fit(X_train_i, y_train_i)

                X_test_i = np.transpose([X_test_i[:, i] for i in feature_indx])
                X_test_i = scaler.transform(X_test_i)

                # evaluate model
                yhat = model.predict(X_test_i)
                # store
                y_true.append(y_test_i[0])
                y_pred.append(yhat[0])

            # calculate accuracy
            acc = accuracy_score(y_true, y_pred)
            accs.append(acc)
        #print(accs)
        return max(accs), accs.index(max(accs))
        
    def predict(self, X):
        # X shape must be (#Trials, #Chans, #Timepoints)
        X_features = np.array([self._getFeatures(eeg_datas) for eeg_datas in X])
        X_features = np.transpose([X_features[:, i] for i in self.feature_indx])
        X_features = self.scaler.transform(X_features)
        return self.model.predict(X_features)

    def predict_proba(self, X):
        # X shape must be (#Trials, #Chans, #Timepoints)
        X_features = np.array([self._getFeatures(eeg_datas) for eeg_datas in X])
        X_features = np.transpose([X_features[:, i] for i in self.feature_indx])
        X_features = self.scaler.transform(X_features)
        return self.model.predict_proba(X_features)

###############################################################################################


class FrameContainer(tk.Tk): 
    def __init__(self, *args, session=None, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        #tk.Tk.iconbitmap(self, default="./clienticon.ico")
        tk.Tk.wm_title(self, "Motor Imagery GUI")
        tk.Tk.minsize(self, 1000, 700)
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        
        frame = StartPage(container, self)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[StartPage] = frame
        
        frame = SessionViewer(container, self)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[SessionViewer] = frame
        
        frame = SessionPrompt(container, self)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[SessionPrompt] = frame

        frame = FeedbackPrompt(container, self)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[FeedbackPrompt] = frame

        frame = TrainingPrompt(container, self)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[TrainingPrompt] = frame
        
        self.current_frame = None

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.show()
        frame.tkraise()
        self.current_frame = frame


class StartPage(tk.Frame):

    def __init__(self, parent, controller, session = None):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Left/Right Hand Motor Imagery Task", font=BOLD_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Start Calibration",
                            command=lambda: controller.show_frame(SessionPrompt))
        button.pack()
        button2 = ttk.Button(self, text="NeuroFeedback",
                            command=lambda: controller.show_frame(FeedbackPrompt))
        button2.pack()
        label2 = tk.Label(self, text="1. Get ready when you hear the beep \n")
        label2.pack(pady=10,padx=10)
        label3 = tk.Label(self, text="2. Imagine the motor movement when you see the arrow pops up \n")
        label3.pack(pady=10,padx=10)
        label4 = tk.Label(self, text="3. Keep imagining until you see the Fixation Cross disappears \n")
        label4.pack(pady=10,padx=10)
        label5 = tk.Label(self, text="4. If no arrow appears, no action is needed \n")
        label5.pack(pady=10,padx=10)

    def show(self):
        return

class SessionViewer(tk.Frame): # Currently for feedback

    def __init__(self, parent, controller, session = None):
        tk.Frame.__init__(self, parent)
        
        self.session = session 
        self.controller = controller

        self.label = tk.Label(self, text="Current Session", font=BOLD_FONT)
        self.label.pack(pady=10,padx=10)

        self.label_avg_focus = tk.Label(self, text="Avg Focus: \t", font=LARGE_FONT)
        self.label_avg_focus.pack(pady=10,padx=10)

        self.label_duration = tk.Label(self, text="Session Duration: \t", font=LARGE_FONT)
        self.label_duration.pack(pady=10,padx=10)
        
        self.button_back = ttk.Button(self, text="Back to NeuroFeedback",
                            command=lambda: self.controller.show_frame(FeedbackPrompt))
        self.button_back.pack()
        
        self.x_values = np.linspace(0, 5, 250 * 5) # replace this in real test

        self.fig = Figure(figsize=(5,5), dpi=100)
        
        self.average_focus_plot = self.fig.add_subplot(2, 1, 1)
        self.average_focus_plot.set_ylim(-2, 2)
        self.average_focus_plot.set_xlim(0, 5) # replace this in real test
        self.average_focus_plot.set_title("Focus Value")
        self.average_focus_plot.set_ylabel("min (-1)   max (1)")
        self.average_focus_plot.set_xticklabels([])
        self.eeg_plot = self.fig.add_subplot(2, 1, 2)
        self.eeg_plot.set_ylim(-400, 400)
        self.eeg_plot.set_xlim(0, 5) # replace this in real test
    
        self.eeg_plot.set_title("EEG Data")
        self.eeg_plot.set_xlabel("Time before now (s)")
        self.eeg_plot.set_ylabel("Voltage (uV)")

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        self.canvas_tk_wid = self.canvas.get_tk_widget()
        self.canvas_tk_wid.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=100)
        
        self.eeg_sampler = EEGSampler()
        with open(Pkl_Filename, 'rb') as file:
            self.model = pickle.load(file) #load the classification model
        self.leftProba = np.array([0.5]*len(self.x_values))
        
        self.update()


    def show(self):
        if self.session != None:
            if self.session.in_session:
                self.button_session.config(text=self.session.session_end_text)
            else : 
                self.button_session.config(text=self.session.session_start_text)
            
    def toggleSession(self):
        self.session.in_session = not self.session.in_session

        if self.session.in_session: 
            # start recording 
            self.session.work_session.start()
           
        else : 
            # end recording 
            self.session.work_session.end()
            # self.ani.event_source.stop()
            # write data 
            self.session.writePastSession(self.session.work_session.getTimestamp(), self.session.work_session.getTotalAvg(), self.session.work_session.getDuration())
        self.show() 
    
    def __plotMultilines(self, ax, xvals, yvals): 
        if ax.lines: 
            #ax.clear()
            for i, line in enumerate(ax.lines):
                line.set_ydata(yvals[i])
        else:
            #ax.clear()
            for i, ys in enumerate(yvals): 
                ax.plot(xvals, ys)
                
    def __plotSingleline(self, ax, xvals, yvals):
        if ax.lines: 
            ax.clear()
            ax.plot(xvals, yvals)
        else:
            ax.clear()
            ax.plot(xvals, yvals)
            
    def update(self):
        self.eeg_sampler.getBuffer()
        self.label.after(4, self.update) # the variable "lable" is a random choice just so "update" can be called again
        
    def animate(self,i):     
                
        xList = self.x_values
        yList = np.flip(self.eeg_sampler.getBuffer().T[0])

        self.__plotSingleline(self.eeg_plot, xList, yList)

        xList = self.x_values
        xInput = self.eeg_sampler.getBuffer()[:1000,:4].T # input for the classifier
        xInput = np.array([xInput]) # reshape
        # print(xInput.shape)
        leftProba = self.model.predict_proba(xInput)[0] # classify the input using the model attribute
        leftProba = leftProba[0] # grab the probability for "left"
        self.leftProba = np.roll(self.leftProba, 1, 0) # append the new probability to the end
        self.leftProba[0] = leftProba
        
        yList = np.flip(self.leftProba)
        self.__plotSingleline(self.average_focus_plot, xList, yList)


class SessionPrompt(tk.Frame):

    def __init__(self, parent, controller, session = None):
        tk.Frame.__init__(self, parent)
        
        self.session = session 
        self.controller = controller

        self.label_timer = tk.Label(self, text = '', font=("Halvetica", 40))
        self.label_timer.pack()
        self.counter = 28800        
        self.label_timer.after(1000, self.update_label_timer)

        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.left_arrowhead = [150+120, 250, 230+120, 300, 230+120, 200]
        self.right_arrowhead = [1000-150-120, 250, 1000-230-120, 300, 1000-230-120, 200]
        self.foot_arrowhead = [500, 500-50, 450, 420-50, 550, 420-50]
        
        # Update
        self.trial_labels = a_123 # the sequence of class labels for the session
        self.trial_index = 0
        self.label_trial = tk.Label(self, text = 'Trial: ' + str(self.trial_index+1) + '/' + str(len(self.trial_labels)),\
                                   font=("Halvetica", 25))
        self.label_trial.pack()

        #TODO: Open Monitor in a new window
        # self.button_back = tk.Button(self, text="Monitor",
        #                     command=lambda: self.controller.show_frame(SessionViewer))
        # self.button_back.pack()

        self.button_skip = tk.Button(self, text="Skip",
                            command=lambda: self.controller.show_frame(FeedbackPrompt))
        self.button_skip.pack()
        self.button_back = tk.Button(self, text="Abort",
                            command=lambda: self.controller.show_frame(StartPage), fg='red')
        self.button_back.pack()

        
        self.trial_time = 0
        pygame.mixer.init()
        pygame.mixer.music.load(BEEP_FILENAME)
        self.can_update = False
        self.can_play = True
        self.render_cross = True
        self.render_left = False
        self.render_right = False
        self.render_foot = False
        self.label_trial.after(1000, self.update)
        self.canvas.after(1000, self.render)
        self.extra_time = 0
        self.extra_time_computed = False
        self.eeg_recorder = EEGRecorder(EEG_OUTPUT_FILENAME) #
        self.event_csv_writer = CSVWriter(EVENT_OUTPUT_FILENAME, column_headers=["timestamp", "event"]) #initializes csv writer
        
        self.last_event_start_recorded = -1
        self.last_event_end_recorded = -1

    # has nothing to do with update function
    # local timer of the clock
    # secondary to understand
    def update_label_timer(self):
        if self.controller.current_frame == self:
            tt = datetime.fromtimestamp(self.counter)
            string = tt.strftime("%H:%M:%S")
            self.label_timer.config(text = string)
            self.counter += 1
            self.can_update = True
        if self.trial_index < len(self.trial_labels):
            self.label_timer.after(1000,self.update_label_timer)
        else:
            self.button_skip.pack_forget()
            
            self.button_back = tk.Button(self, text="Train Model", command=lambda: self.controller.show_frame(TrainingPrompt))
            
            self.button_back.pack()
        
    def update(self):
        if self.controller.current_frame == self and self.can_update:
            # Start EEG recording if it's the first trial
            if self.trial_index == 0 and self.trial_time == 0:
                self.eeg_recorder.start()
            
            # Return early if no more trials
            if self.trial_index >= len(self.trial_labels) : 
                self.eeg_recorder.end() 
                return

            self.trial_time += 125

            if self.trial_time >= START_REST_TIME and self.trial_time < BEEP_FINISH_TIME and self.can_play:
                # Play sound
                pygame.mixer.music.play(loops = 0)
                self.can_play = False
            
            elif self.trial_time >= BEEP_FINISH_TIME and self.trial_time < ARROW_FINISH_TIME:
                # Show arrow
                label = self.trial_labels[self.trial_index]
                self.can_play = True
                if label == 1:
                    self.render_left = True
                    self.render_right = False
                    self.render_foot = False
                elif label == 2:
                    self.render_right = True
                    self.render_left = False
                    self.render_foot = False
                elif label == 3:
                    self.render_foot = True
                    self.render_right = False
                    self.render_left = False
            elif self.trial_time >= ARROW_FINISH_TIME and self.trial_time < TRIAL_FINISH_TIME:
                # Write event start data
                if self.last_event_start_recorded < self.trial_index: 
                    row_data = [time.time()] + ["start_" + str(self.trial_labels[self.trial_index])]
                    self.event_csv_writer.writerow(row_data)
                    self.last_event_start_recorded = self.trial_index
                # Stop rendering arrow and compute random extra time between trials
                self.render_left = False
                self.render_right = False
                self.render_foot = False
                if self.extra_time_computed == False:
                    self.extra_time = random.randint(0,MAX_EXTRA_TIME)
                    self.extra_time_computed = True
            elif self.trial_time >= TRIAL_FINISH_TIME and self.trial_time < TRIAL_BASEEXTRA_TIME + self.extra_time:
                # Write event end data
                if self.last_event_end_recorded < self.trial_index: 
                    row_data = [time.time()] + ["end_" + str(self.trial_labels[self.trial_index])]
                    self.event_csv_writer.writerow(row_data)
                    self.last_event_end_recorded = self.trial_index
                # Stop rendering cross
                self.render_cross = False
            elif self.trial_time >= TRIAL_BASEEXTRA_TIME + self.extra_time:
                # Update trial info
                
                self.trial_index+=1
                self.trial_time = 0
                self.render_cross = True
                self.extra_time_computed = False
                self.label_trial.config(text='Trial: '+str(self.trial_index+1) + '/' + str(len(self.trial_labels)))
        
        self.label_trial.after(125, self.update)

    #Next steps: Get rid of left right down
    # Create green rectangles instead that sits on the horizonal line
    # shifting left and right (on the x-axis)
    # During period where arrow shown, have square shift left/right every 100 ms depending 
    # on number of shift from random number generator 
    def render(self):
        self.canvas.delete('all')
        if self.render_left == True:
            self.canvas.create_polygon(self.left_arrowhead, fill='#1f1', tags='left')
        if self.render_right == True:
            self.canvas.create_polygon(self.right_arrowhead, fill='#1f1', tags='right')
        if self.render_foot == True:
            self.canvas.create_polygon(self.foot_arrowhead, fill='#1f1', tags='foot')
        if self.render_cross == True:
            self.canvas.create_line(1000/2-150+30, 250, 1000/2+150-30, 250, width = 4)
            self.canvas.create_line(500, 150+30, 500, 350-30, width = 4, dash=(4, 4))
        self.canvas.after(50, self.render)
        
    def update1(self):
        if self.controller.current_frame == self:
            self.label_a.config(text = str(self.a))
            self.a -= 1
            if self.a < 0:
                self.a = 3
        if self.a >= 0:
            self.label_a.after(1000,self.update1)
            
    def show(self):
        return

class FeedbackPrompt(tk.Frame):
	#When start button gets clicked everything gets initalized
    def __init__(self, parent, controller, session = None):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="NeuroFeedback", font=BOLD_FONT)
        label.pack(pady=10,padx=10)
        
        self.session = session 
        self.controller = controller

        self.label_timer = tk.Label(self, text = '', font=("Halvetica", 40))
        self.label_timer.pack()
        self.counter = 28800        
        self.label_timer.after(1000, self.update_label_timer)

        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=1)
        
        # Update
         # the sequence of class labels for the session
        self.trial_index = 0

        #TODO: Open Monitor in a new window
        # self.button_back = tk.Button(self, text="Monitor",
        #                     command=lambda: self.controller.show_frame(SessionViewer))
        # self.button_back.pack()

        self.button_monitor = tk.Button(self, text="Stats Monitor",
                            command=lambda: self.controller.show_frame(SessionViewer))
        self.button_monitor.pack()
        self.button_back = tk.Button(self, text="Abort",
                            command=lambda: self.controller.show_frame(StartPage), fg='red')
        self.button_back.pack()
        
        self.trial_time = 0
        pygame.mixer.init()
        pygame.mixer.music.load(BEEP_FILENAME)
        self.can_update = False
        self.can_play = True
        self.render_cross = True
        self.render_left = False
        self.render_right = False
        self.render_foot = False
        # self.label_trial.after(1000, self.update)
        self.canvas.after(1000, self.render)

    def update(self):
    	# Write update rules here
        self.label_trial.after(125, self.update)

    def render(self):
        self.canvas.delete('all')
        # Write elements to render here
        if self.render_cross == True:
            self.canvas.create_line(1000/2-150+30, 250, 1000/2+150-30, 250, width = 4)
            self.canvas.create_line(500, 150+30, 500, 350-30, width = 4, dash=(4, 4))
        self.canvas.after(50, self.render)

    def update_label_timer(self):
        return

    def show(self):
        return

class TrainingPrompt(tk.Frame):
    def __init__(self, parent, controller, session = None):
        tk.Frame.__init__(self, parent)

        self.label = tk.Label(self, text="Training...", font=BOLD_FONT)
        self.label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="NeuroFeedback",
                            command=lambda: controller.show_frame(FeedbackPrompt))
        button.pack()
        self.session = session 
        self.controller = controller
        self.can_train = True
        self.label.after(500, self.try_training)

    def show(self):
        return

    def try_training(self):
        if self.controller.current_frame == self and self.can_train:
            self.train_model()
        self.label.after(500, self.try_training)


    def train_model(self):
        ## Loading data
        eeg_filename = "./data/eeg_data.csv"
        event_filename = "./data/event_data.csv"

        eeg_chans = ['C4','C2', 'C1', 'C3']
        chans = eeg_chans
        eeg_df = pd.read_csv(eeg_filename)
        eeg_df.columns=['time','C4', 'C2', 'C1', 'C3']

        event_df = pd.read_csv(event_filename)
        event_df.columns=['time', 'EventStart']
        event_types = {0:"eye_close", 1:"left", 2:"right", 3:"foot", 4:"idle"}

        # Filter the full data
        filtered_df = eeg_df.copy()
        for chan in chans:
            filtered_df[chan] = filterEEG(filtered_df[chan].values)
        # Process dfs to get labels, raw eeg epochs, epochs of filtered eeg data, filtered epoch data
        output_labels, epoch_times = getOutputLabelsAndEpochTimes(event_df)
        raw_eeg_epochs = getEEGEpochs(epoch_times, eeg_df) # Raw eeg epochs
        filtered_epochs = getEEGEpochs(epoch_times, filtered_df) # Epoched after filtering
        epoch_then_filter_epochs = getFilteredEpochs(raw_eeg_epochs) # Filtered after epoching
        # Create DataFrames
        raw_eeg_epoch_df = getDF(raw_eeg_epochs, output_labels, epoch_times, chans)
        filtered_epoch_df = getDF(filtered_epochs, output_labels, epoch_times, chans)
        epoch_then_filter_df = getDF(epoch_then_filter_epochs, output_labels, epoch_times, chans)
        # Extract trials that are for left vs right hand imagery
        df_to_use = filtered_epoch_df
        filtered_epoch_bi_class_df = df_to_use[(df_to_use['event_type'] == 1) | (df_to_use['event_type'] == 2)]
        X = filtered_epoch_bi_class_df[eeg_chans].values
        Y = filtered_epoch_bi_class_df['event_type'].values
        # Split the 4s trials into 2.8s trials. front and end to obtain 2x trials
        X_split = []
        Y_split = []
        for i in range(len(Y)): 
            eeg_data = []
            for j in range(4): 
                eeg_data.append(X[i, j][:]) # Epoch then filter creates large shifts in voltage near the ends. 
            X_split.append(np.array(eeg_data))
            Y_split.append(Y[i])
        # Shuffle
        X = np.array(X_split)
        Y = np.array(Y_split)
        temp=list(zip(Y, X))
        random.shuffle(temp)
        Y_shuffled, X_shuffled = zip(*temp)

        # Split train/test
        num_train = int(len(X_shuffled)*0.7)
        X_train = np.array(X_shuffled[:num_train])
        X_test = np.array(X_shuffled[num_train:])
        Y_train = np.array(Y_shuffled[:num_train])
        Y_test = np.array(Y_shuffled[num_train:])
        X = np.array(X_shuffled)
        Y = np.array(Y_shuffled)

        # Accuracy on the shuffled train and test split
        num_feats_used = 8
        power_bin_model = PowerBinModel(eeg_chans, num_top=num_feats_used)

        power_bin_model.fit(X_train, Y_train)
        all_feats = power_bin_model.getFeatureNames()
        num_feats = len(all_feats)
        print("num total feats", num_feats)
        print("num used", num_feats_used)
        for i in power_bin_model.feature_indx: 
            print(all_feats[i])

        Y_pred = power_bin_model.predict(X_test)
        print(classification_report(Y_test, Y_pred))
        print("Accuracy:", sum(Y_test==Y_pred) / len(Y_test))

        Pkl_Filename = "PowerBinModel.pkl"

        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(power_bin_model, file)
        
        self.can_train = False

        


if __name__ == "__main__":
    app = FrameContainer(session = None)
    app.mainloop()


# Create a button that will take you to the next window for the neural feedback




