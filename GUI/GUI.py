###############################################################################
# Imports  
###############################################################################

# Matplotlib + FuncAnimation
import matplotlib
matplotlib.use("TkAgg") # Need to set the backend to enable tkinter functionality with matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation

# Tkinter
import tkinter as tk
from tkinter import ttk

# Data model save and loading, 
import pickle
import os

# Misc 
import pygame # For sound play
import webbrowser # For opening Google Maps 
from pyautogui import keyDown, keyUp, click # For controlling Google Maps

# Our own functions
from EEGSampler import EEGSampler
from EEGRecorder import EEGRecorder
from CSVWriter import CSVWriter
from DataProcessingHelper import * 
from PowerBinModel import PowerBinModel
from KeyPress import perform_google_maps_action

# Timing and timers
import time
from datetime import datetime # For creating timer from a counter

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Print a pretty report of the classification success
from sklearn.metrics import classification_report

###############################################################################
# Constants 
###############################################################################

# Recording Settings 
LIVE_DATA = False ## Change this to true if running with live data
CHANNELS_USED = [4, 5, 6, 7] # Corresponding to the pins used on the Cyton

# Calibration Trials
num_trials = 1  # Increase num_trials to 10 or more for live runs. Set to a low number if testing with random data
a1 = [1]*num_trials
a2 = [2]*num_trials
a_123 = a1 + a2
random.shuffle(a_123)

# Neurofeedback Trials 
feedback_num_trials = 2
f1 = [1]*feedback_num_trials
f2 = [2]*feedback_num_trials
f_12 = f1 + f2
random.shuffle(f_12)

# Fonts
BOLD_FONT= ("Verdana", 12, 'bold')

# Filenames
BEEP_FILENAME = './audio/a.mp3'
PRE_RECORDED_EEG_OUTPUT_FILENAME = "./data/eeg_data 15_motorvis.csv" 
PRE_RECORDED_EVENT_OUTPUT_FILENAME = "./data/event_data 15_motorvis.csv" 
LIVE_EEG_OUTPUT_FILENAME = "./data/eeg_data.csv"
LIVE_EVENT_OUTPUT_FILENAME = "./data/event_data.csv"
Pkl_Filename = "PowerBinModel.pkl"

# Trial timing
START_REST_TIME = 1000 
BEEP_FINISH_TIME = START_REST_TIME + 500 # 0.5s to play beep
ARROW_FINISH_TIME = BEEP_FINISH_TIME + 1000 # 1s to show arrow
TRIAL_FINISH_TIME = ARROW_FINISH_TIME + 4000 # 4s for actual trial
TRIAL_BASEEXTRA_TIME = TRIAL_FINISH_TIME + 500 # 0.5s to finish trial
MAX_EXTRA_TIME = 1000 # randomly give an extra 0 to MAX_EXTRA_TRIAL time between trials

###############################################################################
# Classes for frames   
###############################################################################
class FrameContainer(tk.Tk): 
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Motor Imagery GUI")
        tk.Tk.minsize(self, 1000, 700)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.slide_proba = 0.5 # Initialize slider to the center

        with open(Pkl_Filename, 'rb') as file:
            self.model = pickle.load(file) # Load the classification model
        self.eeg_sampler = EEGSampler(LIVE_DATA, channels=CHANNELS_USED, model=self.model)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        frame = StartPage(container, self)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[StartPage] = frame
        
        frame = DataViewer(container, self, self.eeg_sampler)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[DataViewer] = frame
        
        frame = CalibrationPrompt(container, self)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[CalibrationPrompt] = frame

        frame = FeedbackPrompt(container, self, self.eeg_sampler)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[FeedbackPrompt] = frame

        frame = TrainingPrompt(container, self, self.eeg_sampler)
        frame.grid(row=0, column=0, sticky='nsew')
        self.frames[TrainingPrompt] = frame
        
        self.current_frame = None

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        frame.show()
        self.current_frame = frame
    def on_closing(self) :
        self.eeg_sampler.end() # Need to explicitly stop the eeg sampler
        self.destroy()

class StartPage(tk.Frame):

    def __init__(self, parent, controller, session = None):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Left/Right Hand Motor Imagery Task", font=BOLD_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Start Calibration",
                            command=lambda: controller.show_frame(CalibrationPrompt))
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

class DataViewer(tk.Frame): 
    '''
        Visualize the prediction and eeg data
    '''
    def __init__(self, parent, controller, eeg_sampler, session = None):
        tk.Frame.__init__(self, parent)
        
        self.session = session 
        self.controller = controller
        self.eeg_sampler = eeg_sampler

        label = tk.Label(self, text="Data Viewer", font=BOLD_FONT)
        label.pack(pady=10,padx=10)
        
        # Initialize figures and timepoints xvals array
        self.num_seconds_to_display = 5
        self.x_values = np.linspace(0, self.num_seconds_to_display, 250 * self.num_seconds_to_display) # replace this in real test

        self.fig = Figure(figsize=(5,5), dpi=100)
        
        self.average_pred_plot = self.fig.add_subplot(2, 1, 1)
        self.average_pred_plot.set_ylim(0, 1)
        self.average_pred_plot.set_xlim(0, self.num_seconds_to_display) # replace this in real test
        self.average_pred_plot.set_title("Prediction Value")
        self.average_pred_plot.set_ylabel("Probability Left")
        self.average_pred_plot.set_xticklabels([])
        self.eeg_plot = self.fig.add_subplot(2, 1, 2)
        self.eeg_plot.set_ylim(-400, 400)
        self.eeg_plot.set_xlim(0, self.num_seconds_to_display) # replace this in real test
    
        self.eeg_plot.set_title("EEG Data")
        self.eeg_plot.set_xlabel("Time (s)")
        self.eeg_plot.set_ylabel("Voltage (uV)")

        # Create canvas for figures 
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        self.canvas_tk_wid = self.canvas.get_tk_widget()
        self.canvas_tk_wid.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create control buttons
        self.button_back = ttk.Button(self, text="Back to NeuroFeedback",
                    command=lambda: self.controller.show_frame(FeedbackPrompt))
        self.button_back.pack()

        # Only show once even if show() gets called again
        self.in_session = False


    def show(self):
        # Only show once even if show() gets called again
        if not self.in_session: 
            self.in_session = True
            self.ani = animation.FuncAnimation(self.fig, self.animate, interval=100)
            self.canvas.draw()

    def __plotMultilines(self, ax, xvals, yvals): 
        '''
            xvals should be 1d with length of the timepoints 
            yvals can be multiple lines each with same length as xvals
        '''
        if ax.lines: 
            for i, line in enumerate(ax.lines):
                line.set_ydata(yvals[i])
        else:
            for i, ys in enumerate(yvals): 
                ax.plot(xvals, ys)
                
    def animate(self,i):     
        '''
            The animate function for FuncAnimate 
        '''
        xList = self.x_values
        yList = self.eeg_sampler.getBuffer()[200:len(xList) + 200, :4].T # Offset by 100 to reduce the visibility of the filter lag
        yFlipped = []
        for elem in yList:
            yFlipped.append(np.flip(elem))
        self.__plotMultilines(self.eeg_plot, xList, yFlipped)

        xList = self.x_values
        yList = np.flip(self.eeg_sampler.getPredictionValues().T[100:len(xList) + 100])
        self.__plotMultilines(self.average_pred_plot, xList, [yList])


class CalibrationPrompt(tk.Frame):

    def __init__(self, parent, controller, session = None):
        tk.Frame.__init__(self, parent)
        
        self.session = session 
        self.controller = controller

        # The sequence of class labels for the session
        self.trial_labels = a_123 

        # Timer label
        self.label_timer = tk.Label(self, text = '', font=("Halvetica", 40))
        self.label_timer.pack()

        # Create cross canvas 
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=1)

        # Initialize arrows dimensions
        self.left_arrowhead = [150+120, 250, 230+120, 300, 230+120, 200]
        self.right_arrowhead = [1000-150-120, 250, 1000-230-120, 300, 1000-230-120, 200]
        
        # Trial label
        self.label_trial = tk.Label(self, text = 'Trial: 1/' + str(len(self.trial_labels)),\
                    font=("Halvetica", 25))
        self.label_trial.pack()

        # Define buttons that may be used
        self.button_skip = None
        self.button_back = None
        self.button_train = None

        # Load the beep noise
        pygame.mixer.init()
        pygame.mixer.music.load(BEEP_FILENAME)

        # Set the update mode to false
        self.can_update = False
        

    def show(self): 
        if not self.can_update: 
            self.can_update = True
            self.can_play = True
            
            # Initialize labels and buttons
            self.label_trial.config(text = 'Trial: 1/' + str(len(self.trial_labels)))

            self.button_skip = tk.Button(self, text="Skip", command=self.abortAndSkip)
            self.button_skip.pack()

            self.button_back = tk.Button(self, text="Abort", command=self.abortAndGoToStart, fg='red')
            self.button_back.pack()

            # Initialize recording streams
            self.eeg_recorder = EEGRecorder(LIVE_DATA, LIVE_EEG_OUTPUT_FILENAME, channels=CHANNELS_USED)
            self.event_csv_writer = CSVWriter(LIVE_EVENT_OUTPUT_FILENAME, column_headers=["timestamp", "event"]) #initializes csv writer
            
            # Initialize the event recording variables
            self.trial_index = 0
            self.trial_time = 0
            self.extra_time = 0
            self.extra_time_computed = False
            self.render_cross = True
            self.render_left = False
            self.render_right = False
            self.last_event_start_recorded = -1
            self.last_event_end_recorded = -1

            # Begin the update loops for timer and prompt
            self.counter = 28800        
            self.label_timer.after(100, self.update_label_timer)
            self.label_trial.after(100, self.update)
            self.canvas.after(100, self.render)
    
    ####################
    ## UPDATE METHODS ##
    ####################

    # Update loop for timer
    def update_label_timer(self):
        if self.controller.current_frame == self and self.can_update:
            tt = datetime.fromtimestamp(self.counter)
            string = tt.strftime("%H:%M:%S")
            self.label_timer.config(text = string)
            self.counter += 1
        if self.trial_index < len(self.trial_labels) and self.can_update:
            self.label_timer.after(1000,self.update_label_timer)
    
    # Update loop for prompt
    def update(self):
        if self.controller.current_frame == self and self.can_update:
            # Start EEG recording if it's the first trial
            if self.trial_index == 0 and self.trial_time == 0:
                self.eeg_recorder.start()
            
            # Return early if no more trials
            if self.trial_index >= len(self.trial_labels) : 
                self.eeg_recorder.end() 
                self.button_skip.pack_forget()

                self.button_train = tk.Button(self, text="Train Model", command=self.abortAndTrain)
                
                self.button_train.pack()
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
                elif label == 2:
                    self.render_right = True
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
                if self.trial_index < len(self.trial_labels):
                    self.label_trial.config(text='Trial: '+str(self.trial_index+1) + '/' + str(len(self.trial_labels)))
                else :
                    self.label_trial.config(text='Trials Completed! Please "Train Model" to continue')
            self.label_trial.after(125, self.update)

    # Create green rectangles instead that sits on the horizonal line
    # shifting left and right (on the x-axis)
    # During period where arrow shown, have square shift left/right every 100 ms depending 
    # on number of shift from random number generator 
    def render(self):
        if self.can_update:
            self.canvas.delete('all')
            if self.render_left == True:
                self.canvas.create_polygon(self.left_arrowhead, fill='#1f1', tags='left')
            if self.render_right == True:
                self.canvas.create_polygon(self.right_arrowhead, fill='#1f1', tags='right')
            if self.render_cross == True:
                self.canvas.create_line(1000/2-150+30, 250, 1000/2+150-30, 250, width = 4)
                self.canvas.create_line(500, 150+30, 500, 350-30, width = 4, dash=(4, 4))
            self.canvas.after(50, self.render)

    
    ###################
    ## ABORT METHODS ##
    ###################
    def abort(self):
        print("Abort calibration prompt")
        if self.can_update:
            # Stop the EEG recorder if aborting
            self.eeg_recorder.end()

            # Clear the buttons because they will be created again if returned to calibration prompt
            if self.button_back: 
                self.button_back.pack_forget()
            if self.button_skip: 
                self.button_skip.pack_forget()
            if self.button_train: 
                self.button_train.pack_forget()
            self.can_update = False

    def abortAndSkip(self): 
        self.abort()
        self.controller.show_frame(FeedbackPrompt)

    def abortAndGoToStart(self):
        self.abort()
        self.controller.show_frame(StartPage)

    def abortAndTrain(self):
        self.abort()
        self.controller.show_frame(TrainingPrompt)
            

class TrainingPrompt(tk.Frame):
    def __init__(self, parent, controller, eeg_sampler, session = None):
        tk.Frame.__init__(self, parent)

        self.controller = controller
        self.eeg_sampler = eeg_sampler

        self.label = tk.Label(self, text="Training...", font=BOLD_FONT)
        self.label.pack(pady=10,padx=10)
        self.model_trained = False


    def show(self):
        if not self.model_trained:
            self.train_model()

            # Update the EEG sampler's model with the newly trained
            with open(Pkl_Filename, 'rb') as file:
                model = pickle.load(file) 
                self.eeg_sampler.setModel(model)
            
            self.model_trained = True

    def train_model(self):

        start_time = time.time()

        ## Loading data
        eeg_filename = LIVE_EEG_OUTPUT_FILENAME if LIVE_DATA else PRE_RECORDED_EEG_OUTPUT_FILENAME
        event_filename =  LIVE_EVENT_OUTPUT_FILENAME if LIVE_DATA else PRE_RECORDED_EVENT_OUTPUT_FILENAME

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
        filtered_epochs = getEEGEpochs(epoch_times, filtered_df, eeg_chans) # Epoched after filtering

        # Create DataFrames
        filtered_epoch_df = getDF(filtered_epochs, output_labels, epoch_times, chans)
        # Extract trials that are for left vs right hand imagery
        df_to_use = filtered_epoch_df
        filtered_epoch_bi_class_df = df_to_use[(df_to_use['event_type'] == 1) | (df_to_use['event_type'] == 2)]
        X = filtered_epoch_bi_class_df[eeg_chans].values
        Y = filtered_epoch_bi_class_df['event_type'].values

        # Accuracy on the shuffled train and test split
        num_feats_used = 8
        power_bin_model = PowerBinModel(eeg_chans, num_top=num_feats_used)

        power_bin_model.fit(X, Y)
        all_feats = power_bin_model.getFeatureNames()
        num_feats = len(all_feats)
        print("num total feats", num_feats)
        print("num used", num_feats_used)
        for i in power_bin_model.feature_indx: 
            print(all_feats[i])

        with open(Pkl_Filename, 'wb') as file:  
            pickle.dump(power_bin_model, file)

        end_time = time.time()
        print("finished training")
        self.label.config(text = "Finished Training in " + str(end_time-start_time) + " seconds!")
        self.button = ttk.Button(self, text="Continue to NeuroFeedback",
                            command=lambda: self.controller.show_frame(FeedbackPrompt))
        self.button.pack()


class FeedbackPrompt(tk.Frame):
	#When start button gets clicked everything gets initalized
    def __init__(self, parent, controller, eeg_sampler, session = None):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="NeuroFeedback", font=BOLD_FONT)
        label.pack(pady=10,padx=10)
        
        self.session = session 
        self.controller = controller

        # The sampler from EEGSampler has a getPredictionValue() method that you can use to get a % value for left
        self.eeg_sampler = eeg_sampler

        # Create timer label
        self.label_timer = tk.Label(self, text = '', font=("Halvetica", 40))
        self.label_timer.pack()

        # Create a static instructions label
        self.instructions_label = tk.Label(self, text = '')
        self.instructions_label.pack()

        # Create cross canvas
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=1)

        # Initialize arrows dimensions
        self.left_arrowhead = [150+120, 250, 230+120, 300, 230+120, 200]
        self.right_arrowhead = [1000-150-120, 250, 1000-230-120, 300, 1000-230-120, 200]
        
        # Trial label
        self.trial_labels = f_12
        self.label_trial = tk.Label(self, text = 'Trial: 1/' + str(len(self.trial_labels)),\
                    font=("Halvetica", 25))
        self.label_trial.pack()
        
        # Create the control buttons
        self.button_monitor = None
        self.button_back = None
        self.button_use = None
        
        # Load beep noises 
        pygame.mixer.init()
        pygame.mixer.music.load(BEEP_FILENAME)
        
        # Initialize to not be running
        self.can_update = False

    ##################
    ## SHOW METHODS ##
    ##################
    def show(self):
        if not self.can_update: 
            self.can_update = True

            self.instructions_label.config(text='Please wait for the beep to begin neurofeedback session (around 5 seconds)')
            self.label_trial.config(text = 'Trial: 1/' + str(len(self.trial_labels)))
            
            # Create the control buttons
            self.button_monitor = tk.Button(self, text="Stats Monitor",
                                command=lambda: self.controller.show_frame(DataViewer))
            self.button_monitor.pack()
            self.button_back = tk.Button(self, text="Abort",
                                command=self.abortAndGoToStart, fg='red')
            self.button_back.pack()
            self.button_use = None

            # Start sampling 
            self.eeg_sampler.startStream()

            # Initialize some variables that may be helpful
            self.trial_index = 0
            self.trial_time = 0
            self.can_play = True
            self.render_cross = True
            self.render_left = False
            self.render_right = False
            self.extra_time_computed = False
            self.extra_time = 0

            # Begin the update loops for timer and prompt
            self.counter = 28800 
            self.slide_proba = 0.5 # Initialize the slider to the center
            self.label_timer.after(100, self.update_label_timer)
            self.label_trial.after(5000, self.startUpdate)
            self.canvas.after(100, self.render)
            
    
    ####################
    ## UPDATE METHODS ##
    ####################
    def update_label_timer(self):
        if self.can_update:
            tt = datetime.fromtimestamp(self.counter)
            string = tt.strftime("%H:%M:%S")
            self.label_timer.config(text = string)
            self.counter += 1
        if self.trial_index < len(self.trial_labels) and self.can_update:
            self.label_timer.after(1000,self.update_label_timer)

    def startUpdate(self): 
        if self.can_update: 
            self.instructions_label.config(text="Try to get the slider to the left or right side by thinking left or right!")
            self.update()

    def update(self):
    	# Write update rules here
        if self.can_update: 
            
            # Return early if no more trials
            if self.trial_index >= len(self.trial_labels) and self.button_use is None: 
                self.button_use = tk.Button(self, text="Launch Google Maps", command=self.useApp)
                self.button_use.pack()

            self.trial_time += 125
            # Only provide prompts and trial count if the use button is not shown
            if self.button_use is None: 
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
                    elif label == 2:
                        self.render_right = True
                        self.render_left = False
                elif self.trial_time >= ARROW_FINISH_TIME and self.trial_time < TRIAL_FINISH_TIME:
                    # Stop rendering arrow and compute random extra time between trials
                    self.render_left = False
                    self.render_right = False
                    if self.extra_time_computed == False:
                        self.extra_time = random.randint(0,MAX_EXTRA_TIME)
                        self.extra_time_computed = True
                elif self.trial_time >= TRIAL_FINISH_TIME and self.trial_time < TRIAL_BASEEXTRA_TIME + self.extra_time:
                    # Keep rendering cross
                    self.render_cross = True
                elif self.trial_time >= TRIAL_BASEEXTRA_TIME + self.extra_time:
                    # Update trial info
                    self.trial_index+=1
                    self.trial_time = 0
                    self.render_cross = True
                    self.extra_time_computed = False
                    if self.trial_index < len(self.trial_labels):
                        self.label_trial.config(text='Trial: '+str(self.trial_index+1) + '/' + str(len(self.trial_labels)))
                    else :
                        self.label_trial.config(text='NeuroFeedback Completed! Please "Launch Google Maps" to continue')

            self.slide_proba = self.eeg_sampler.getPredictionValue() # Represents left probability
            self.label_timer.after(125, self.update)

    def useApp(self):
        # Launch Google Maps 
        print("loading browser")
        webbrowser.open('https://www.google.com/maps/@32.8824001,-117.2401516,3a,75y,350.34h,92.6t/data=!3m6!1e1!3m4!1stk9EAp1VOrQ_dceJZFAYAg!2e0!7i16384!8i8192', new=2)
        self.button_use.pack_forget()
        self.label_trial.config(text='Google Maps Launched!')
        time.sleep(3)
        print("input starting in: ")
        time.sleep(1)
        print("5")
        time.sleep(1)
        print("4")
        time.sleep(1)
        print("3")
        click(600, 600) # Click the window to enable arrow key inputs
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)
        print("go!!")
        
        # Send commands 
        self.sendCommands()

    def sendCommands(self):
        if self.can_update:
            perform_google_maps_action((1 - self.slide_proba) * 2 - 1)
            self.label_trial.after(250, self.sendCommands)

    def render(self):
        if self.can_update: 
            self.canvas.delete('all')
            # Write elements to render here
            if self.render_left == True:
                self.canvas.create_polygon(self.left_arrowhead, fill='#1f1', tags='left')
            if self.render_right == True:
                self.canvas.create_polygon(self.right_arrowhead, fill='#1f1', tags='right')
            if self.render_cross == True:
                bar_loc = (1 - self.slide_proba) * 240 # Need to flip slide_proba to have a slider on the right correspond with high right proba
                self.canvas.create_line(1000/2-150+30, 250, 1000/2+150-30, 250, width = 4)
                self.canvas.create_line(500, 150+30, 500, 350-30, width = 4, dash=(4, 4))
                self.canvas.create_polygon([492-120+bar_loc, 290, 492-120+bar_loc, 210, 508-120+bar_loc, 210, 508-120+bar_loc, 290], fill='grey', tags='slide-bar')
            self.canvas.after(100, self.render)

    ###################
    ## ABORT METHODS ##
    ###################
    def abort(self):
        if self.can_update: 
            # Clear the buttons because they will be created again if returned to calibration prompt
            if self.button_back: 
                self.button_back.pack_forget()
            if self.button_monitor: 
                self.button_monitor.pack_forget()
            if self.button_use: 
                self.button_use.pack_forget()
            # End sampling
            self.eeg_sampler.end()
            self.can_update = False

    def abortAndGoToStart(self):
        self.abort()
        self.controller.show_frame(StartPage)



###############################################################################
# Main  
###############################################################################
if __name__ == "__main__":
    app = FrameContainer()
    app.mainloop()




