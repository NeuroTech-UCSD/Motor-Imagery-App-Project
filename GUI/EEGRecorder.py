import numpy as np
from CSVWriter import CSVWriter

import time
from datetime import datetime

import threading
import atexit

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) # uV/count
DEFAULT_FS = 250
DEFAULT_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7] # All 8 channels of cyton

class EEGRecorder: 
    def __init__(self, live, output_filename, channels=DEFAULT_CHANNELS):
        self.fs = DEFAULT_FS
        self.output_filename = output_filename
        self.started = False
        self.live = live
        self.channels = channels
        atexit.register(self.end)
    
    ############################
    ## STREAM CONTROL METHODS ##
    ############################
    def start(self):
        print("start recording called")
        if self.live: 
            from pyOpenBCI import OpenBCICyton
            print("LIVE: started eeg recording")
            header = ["timestamp"] + self.channels
            self.csv_writer = CSVWriter(self.output_filename, column_headers=header)
            self.board = OpenBCICyton()
            self.eeg_thread = threading.Thread(target=self.board.start_stream, args=(self.record_data_sample,))
            self.eeg_thread.start()
            self.started = True

    def end(self): 
        print("end recording called")
        if self.started: 
            print("LIVE: ended eeg recording")
            if self.live: 
                self.board.stop_stream()
            self.started = False

    ####################
    ## HELPER METHODS ##
    ####################
    def record_data_sample(self, sample):
        # Get timestamp
        now = time.time()
        
        # Get the scaled channel data
        if self.live: 
            raw_eeg_data = np.array(sample.channels_data) * SCALE_FACTOR_EEG
        else :
            raw_eeg_data = np.array(sample) 
        
        # Record to CSV
        row_data = [now]
        row_data.extend(raw_eeg_data[self.channels])
        self.csv_writer.writerow(row_data)



    