# Calibration GUI
Motor Imagery Prompts for Left, Right, Foot, and Idle. 

## USAGE
python CalibrationGUI.py 
- make sure to change LIVE_DATA in EEGRecorder to false if not recording EEG
- make sure to have a ./data/ folder created

## Pins 
| Pin  | 10/20 Location | Function  |
|:-----|:---------|:----------|
| Bias | Fpz | Ground |
| N8P | C3 | Left-most channel |
| N7P | C1 | Left-center channel |
| N6P | C2 | Right-center channel  |
| N5P | C4 | Right-most channel |
| SRB | Earclip (left) | Reference |
