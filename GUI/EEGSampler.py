import time
from datetime import datetime

import numpy as np
import pyeeg
import threading
import random

from PowerBinModel import PowerBinModel
from DataProcessingHelper import filterEEG
import atexit


SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) # uV/count

DEFAULT_PREDICTION_WINDOW_SECONDS = 4 # Window in seconds needed for prediction
DEFAULT_UPDATE_SECONDS = 0.5 # Number of seconds to wait before updating prediction values

DEFAULT_SECONDS = 10 # 10 second long buffer 
DEFAULT_FS = 250
DEFAULT_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7] # These are all the Cyton channels
        
class EEGSampler:
    """ Holds a buffer of EEG data and accepts a single data sample as input to append to the buffer
    """
    def __init__(self, live, fs=DEFAULT_FS, buffer_seconds=DEFAULT_SECONDS, update_seconds=DEFAULT_UPDATE_SECONDS, prediction_seconds=DEFAULT_PREDICTION_WINDOW_SECONDS, channels=DEFAULT_CHANNELS, model=None, to_clean=True):
        self.fs = fs
        self.buffer_seconds = buffer_seconds
        self.prediction_seconds = prediction_seconds
        self.timepoints_to_chop = 300 # How many timepoints to disregard from the most recent (this sets back the lag of prediction)
        if to_clean: 
            if prediction_seconds * fs > (buffer_seconds * fs - self.timepoints_to_chop): 
                print("Warning: prediction window may contain edge artifacts because it is larger than 500 timepoints less than the buffer size")
                self.timepoints_to_chop = (buffer_seconds * fs - prediction_seconds * fs) // 2 # Take the middle half to reduce edge artifacts
        self.channels = channels # These are the channels we want
        
        self.live = live
        self.model = model
        if model is not None: 
            self.classes = list(self.model.model.classes_)
            self.index_of_left = self.classes.index(1)
            print("classes", self.classes)

        # If generating artificial data, this will cap the # of seconds we get artificial data to prevent while(true) runaways
        self.seconds_to_gather_artificial_data = 600 
        
        self.buffer = np.zeros((fs * buffer_seconds, len(channels)))
        self.raw_buffer = np.zeros((fs * buffer_seconds, len(channels)))
        self.dc_removed_buffer = np.zeros((fs * buffer_seconds, len(channels)))
        
        self.pred_values = np.zeros((fs * buffer_seconds))
        self.pred_values_average = np.zeros((fs * buffer_seconds))
        self.update_seconds = update_seconds
        self.last_updated = time.time()
        self.mean = np.zeros(len(channels))
        self.count_samples = 0
        self.to_clean = to_clean
        self.started = False
        atexit.register(self.end)
    
    ############################
    ## STREAM CONTROL METHODS ##
    ############################
    def setModel(self, model):
        '''
            Allows for the model to be updated
        '''
        self.model = model
        self.classes = list(self.model.model.classes_)
        self.index_of_left = self.classes.index(1)
        print("classes", self.classes)

    def startStream(self) :
        print("start streaming called")
        if not self.started:
            self.started = True

            if self.live: 
                from pyOpenBCI import OpenBCICyton
                print("LIVE: started eeg streaming")
                self.board = OpenBCICyton()
                self.eeg_thread = threading.Thread(target=self.board.start_stream, args=(self.push_data_sample,))
            else :
                print("FAKE DATA: started")
                self.eeg_thread = threading.Thread(target=self.__generate_artificial_data, args=(self.push_data_sample,))
            self.eeg_thread.start()

    def end(self): 
        print("end stream called")
        if self.started: 
            self.started = False
            if self.live: 
                print("LIVE: ended eeg streaming")
                self.board.stop_stream()
            else :
                print("FAKE DATA: ended")
    
    ########################
    ## BUFFER GET METHODS ##
    ########################
    
    def getBuffer(self):
        return self.buffer
        
    def getPredictionValue(self):  
        return self.pred_values[0]
    def getPredictionValues(self):  
        return self.pred_values

    def getPredictionValueAverage(self):  
        return self.pred_values_average[0]
    def getPredictionValueAverages(self):  
        return self.pred_values_average

    
    ############################
    ## PRIVATE HELPER METHODS ##
    ############################
    def __generate_artificial_data(self, callback) : 
        limit_counter = 0 # This will prevent this thread from continuing to produce values on while forever
        while self.started and (limit_counter < (600 * self.fs)): # TODO: remove 
            fake_sample = np.random.rand(8) * 300 # Cyton has 8 channels by default, so we generate a value for each. 
            callback(fake_sample)
            time.sleep(0.004) # Sample at 250 Hz 
            limit_counter += 1
        
    def __filter_eeg(self):
        # Bandpass + 60 Hz Notch
        for i, chan in enumerate(self.channels):
            self.buffer[:self.count_samples, i] = filterEEG(self.dc_removed_buffer[:self.count_samples, i], self.fs, (0.5, 50))

    def __update_prediction_buffer(self):
        # Take a 4 second window timepoints_to_chop timepoints away from the end of the buffer to reduce edge effects
        data = np.transpose(self.buffer[self.timepoints_to_chop : self.prediction_seconds * self.fs + self.timepoints_to_chop])
        
        prediction = self.model.predict_proba(np.array([data]))[0]
        print("prediction: ", prediction)
        prediction = prediction[self.index_of_left]
        self.pred_values[0] = prediction 
        
        self.pred_values_average = np.roll(self.pred_values_average, 1, 0) 
        # Update average values
        self.pred_values_average[0] = np.mean(self.pred_values[:self.fs])

    
    ###########################
    ## CALLBACK FOR NEW DATA ##
    ###########################
    def push_data_sample(self, sample):
        # Count the number of samples up till the full buffer (this is for mean calculation)
        if self.count_samples < self.fs * self.buffer_seconds: 
            self.count_samples += 1
        
        # Get the scaled channel data
        if self.live: 
            raw_eeg_data = np.array(sample.channels_data)[self.channels] * SCALE_FACTOR_EEG
        else :
            raw_eeg_data = np.array(sample)[self.channels]
        
        # Roll and prepend the buffer with the new data
        self.raw_buffer = np.roll(self.raw_buffer, 1, 0)
        self.buffer = np.roll(self.buffer, 1, 0)
        self.dc_removed_buffer = np.roll(self.dc_removed_buffer, 1, 0)
        self.pred_values = np.roll(self.pred_values, 1, 0)
        self.pred_values_average = np.roll(self.pred_values_average, 1, 0)
        for i, chan in enumerate(self.channels):
            self.raw_buffer[0, i] = raw_eeg_data[i]
            if self.to_clean:  
                self.dc_removed_buffer[0, i] = self.raw_buffer[0,i] - self.mean[i]
                self.buffer[0, i] = self.dc_removed_buffer[0, i]
            else : 
                self.dc_removed_buffer[0, i] = self.raw_buffer[0, i] 
                self.buffer[0, i] = self.raw_buffer[0, i] 
            self.pred_values[0] = self.pred_values[1]
            self.pred_values_average[0] = np.mean(self.pred_values[:self.fs])
        
        # Calculate the new mean if the update time has passed
        now = time.time()
        if (now - self.last_updated) > self.update_seconds :
            self.last_updated = now

            # Only predict once we've reached the number of samples we need in the buffer
            if (self.count_samples > self.prediction_seconds * self.fs + self.timepoints_to_chop): 
                if (self.to_clean):
                    self.__filter_eeg()
                if (self.model is not None):
                    self.__update_prediction_buffer()
   
                