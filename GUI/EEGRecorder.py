from pyOpenBCI import OpenBCICyton
import numpy as np
from CSVWriter import CSVWriter

import time
from datetime import datetime

import threading
import atexit

SCALE_FACTOR_EEG = (4500000)/24/(2**23-1) #uV/count
DEFAULT_FS = 250
CYTON_CHANNELS = [4, 5, 6, 7]

LIVE_DATA = False

class EEGRecorder: 
    def __init__(self, output_filename):
        self.fs = DEFAULT_FS
        self.output_filename = output_filename
        self.started = False
        self.live = LIVE_DATA
        atexit.register(self.end)

    
    def start(self):
        print("started eeg recording")
        header = ["timestamp"] + CYTON_CHANNELS
        self.csv_writer = CSVWriter(self.output_filename, column_headers=header)
        if self.live: 
            self.board = OpenBCICyton()
            self.eeg_thread = threading.Thread(target=self.board.start_stream, args=(self.record_data_sample,))
            self.eeg_thread.start()
        self.started = True
        
    
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
        row_data.extend(raw_eeg_data[CYTON_CHANNELS])
        self.csv_writer.writerow(row_data)


    def end(self): 
        if self.started: 
            print("ended eeg recording")
            if self.live: 
                self.board.stop_stream()
            self.started = False
    