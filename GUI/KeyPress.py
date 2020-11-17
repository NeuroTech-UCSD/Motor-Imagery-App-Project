# For type enforcement
from typing import List


'''

	Usage: for default, input [-1,1] left_probability to determine and send keypress to window


	Param	   | Type				| Def
	_________________________________________________________________________
	left_prob  | float				| Probability of left event
	prob_range | List[float]		| Range for left_prob
	thresholds | List[List[float]]	| Specific range for each keypress event
	keys       | List[str]			| Keys to press

'''
def perform_google_maps_action(
	left_prob: float,
	prob_range:List[float] = [-1,1],
	thresholds:List[List[float]] = [[-1,-1/3],[-1/3,1/3],[1/3,1]],
	keys:List[str] = ["left","up","right"]
):
	# Making sure params are correctly used
	assert (len(prob_range) == 2)
	assert (prob_range[1] > prob_range[0])
	assert (left_prob >= prob_range[0])
	assert (left_prob <= prob_range[1])

	assert (len(thresholds) == len(keys))
	
	for t in thresholds:
		assert len(t) == 2 and t[0] < t[1]

	for i in range(len(thresholds)-1):
		assert thresholds[i][1] == thresholds[i+1][0]

	# Find which key we're pressing
	keypress_ind = 0
	for i, t in enumerate(thresholds):
		if left_prob >= t[0] and left_prob <= t[1]:
			keypress_ind = i

	# Send keypress to active window
	send_keypress(keys[keypress_ind])

'''

	Usage: should only be called through perform_google_maps_action


	Param	   | Type				| Def
	_________________________________________________________________________
	key        | str				| Key to press

'''

# Requires tkinter, install pyautogui via pip then follow instructions on runtime
from pyautogui import keyDown, keyUp, click

# Cosmetic
from datetime import datetime

import time
import numpy as np

def send_keypress(key: str):
	# Header for event
	print(datetime.now().strftime("%m/%d/%Y %H:%M:%S").center(40,"="))
	print(f"Got a call for '{key}'.")

	# Send Keypress
	keyDown(key)
	print(f"Sent keypress for '{key}'.")
	time.sleep(0.5)
	keyUp(key)

	# Footer for event
	print("="*40,end="\n\n")


if __name__ == "__main__":
	import webbrowser
	webbrowser.open('https://www.google.com/maps/@32.8824001,-117.2401516,3a,75y,350.34h,92.6t/data=!3m6!1e1!3m4!1stk9EAp1VOrQ_dceJZFAYAg!2e0!7i16384!8i8192', new=2)
	print("loading browser")
	time.sleep(3)
	print("input starting in: ")
	time.sleep(1)
	print("5")
	time.sleep(1)
	print("4")
	time.sleep(1)
	print("3")
	click(400, 200)
	time.sleep(1)
	print("2")
	time.sleep(1)
	print("1")
	time.sleep(1)
	print("go!!")
	inputs = (np.random.rand(20) * 2) - 1
	print(inputs)
	for i in inputs:
		time.sleep(0.25)
		perform_google_maps_action(i)


