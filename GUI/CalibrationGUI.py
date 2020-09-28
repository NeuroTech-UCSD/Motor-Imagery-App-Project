import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation
import os
from datetime import datetime 
import pygame
import random
from EEGRecorder import EEGRecorder
from CSVWriter import CSVWriter
import time
import numpy as np

# Trials
num_trials = 3
a1 = [1]*num_trials
a2 = [2]*num_trials
a3 = [3]*num_trials
a4 = [4]*num_trials
a_123 = a1 + a2 + a3 + a4
random.shuffle(a_123)

# Fonts
LARGE_FONT= ("Verdana", 12)
BOLD_FONT= ("Verdana", 12, 'bold')
UNDERLINE_FONT = ("Verdana", 12, 'underline')

# Filenames
BEEP_FILENAME = './audio/a.mp3'
EEG_OUTPUT_FILENAME = "./data/eeg_data.csv"
EVENT_OUTPUT_FILENAME = "./data/event_data.csv"

# Trial timing
START_REST_TIME = 1000 
BEEP_FINISH_TIME = START_REST_TIME + 500 # 0.5s to play beep
ARROW_FINISH_TIME = BEEP_FINISH_TIME + 1000 # 1s to show arrow
TRIAL_FINISH_TIME = ARROW_FINISH_TIME + 4000 # 4s for actual trial
TRIAL_BASEEXTRA_TIME = TRIAL_FINISH_TIME + 500 # 0.5s to finish trial
MAX_EXTRA_TIME = 1000 # randomly give an extra 0 to MAX_EXTRA_TRIAL time between trials

class FrameContainer(tk.Tk): 
    def __init__(self, *args, session=None, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        #tk.Tk.iconbitmap(self, default="./clienticon.ico")
        tk.Tk.wm_title(self, "Calibration Prompt")
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
        label = tk.Label(self, text="Calibration Session", font=BOLD_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Start",
                            command=lambda: controller.show_frame(SessionPrompt))
        button.pack()

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

class SessionViewer(tk.Frame):

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

        if self.session != None:
            self.button_session = ttk.Button(self, text=self.session.session_start_text,
                                command=self.toggleSession)
            self.button_session.pack()
        
        self.button_back = ttk.Button(self, text="Back to Prompt",
                            command=lambda: self.controller.show_frame(SessionPrompt))
        self.button_back.pack()
        
        if self.session != None:
            self.x_values = np.linspace(0, self.session.work_session.buffer_seconds, self.session.work_session.fs * self.session.work_session.buffer_seconds)

        self.fig = Figure(figsize=(5,5), dpi=100)
        
        self.average_focus_plot = self.fig.add_subplot(2, 1, 1)
        self.average_focus_plot.set_ylim(-2, 2)
        if self.session != None:
            self.average_focus_plot.set_xlim(0, self.session.work_session.buffer_seconds)
        self.average_focus_plot.set_title("Focus Value")
        self.average_focus_plot.set_ylabel("min (-1)   max (1)")
        self.average_focus_plot.set_xticklabels([])
        self.eeg_plot = self.fig.add_subplot(2, 1, 2)
        self.eeg_plot.set_ylim(-400, 400)
        if self.session != None:
            self.eeg_plot.set_xlim(0, self.session.work_session.buffer_seconds)
        self.eeg_plot.set_title("EEG Data")
        self.eeg_plot.set_xlabel("Time before now (s)")
        self.eeg_plot.set_ylabel("Voltage (uV)")

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        self.canvas_tk_wid = self.canvas.get_tk_widget()
        self.canvas_tk_wid.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.ani = animation.FuncAnimation(self.fig, self.animate, interval=100)

        # canvas2 = FigureCanvasTkAgg(f2, self)
        # canvas2.draw()
        # canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        # toolbar = NavigationToolbar2Tk(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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
        
    def animate(self, i):     
        if self.session != None:
            if self.session.work_session.started: 
                xList = self.x_values
                yList = np.transpose(self.session.work_session.getEEGBufferData())

                self.__plotMultilines(self.eeg_plot, xList, yList)

                xList = self.x_values
                yList = np.array([self.session.work_session.getAveragedFocusBufferData()])
                self.__plotMultilines(self.average_focus_plot, xList, yList)

                self.label_avg_focus.config(text="Avg Focus: " + '{:.4f}'.format(self.session.work_session.getTotalAvg()))
                self.label_duration.config(text="Session Duration: " + self.session.work_session.getDuration())

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
        
        self.trial_labels = a_123 # the sequence of class labels for the session
        self.trial_index = 0
        self.label_trial = tk.Label(self, text = 'Trial: ' + str(self.trial_index+1) + '/' + str(len(self.trial_labels)),\
                                   font=("Halvetica", 25))
        self.label_trial.pack()

        #Open this in a new window
        self.button_back = tk.Button(self, text="Monitor",
                            command=lambda: self.controller.show_frame(SessionViewer))
        self.button_back.pack()
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
        self.eeg_recorder = EEGRecorder(EEG_OUTPUT_FILENAME)
        self.event_csv_writer = CSVWriter(EVENT_OUTPUT_FILENAME, column_headers=["timestamp", "event"])
        
        self.last_event_start_recorded = -1
        self.last_event_end_recorded = -1

    def update_label_timer(self):
        if self.controller.current_frame == self:
            tt = datetime.fromtimestamp(self.counter)
            string = tt.strftime("%H:%M:%S")
            self.label_timer.config(text = string)
            self.counter += 1
            self.can_update = True
        self.label_timer.after(1000,self.update_label_timer)
        
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

if __name__ == "__main__":
    app = FrameContainer(session = None)
    app.mainloop()
