# Calibration GUI
Motor Imagery Prompts for Left, Right, Foot, and Idle. 

## USAGE
python CalibrationGUI.py 
- make sure to change LIVE_DATA in EEGRecorder to false if not recording EEG
- make sure to have a ./data/ folder created

## Pins (Cyton)
| Pin  | 10/20 Location | Function  |
|:-----|:---------|:----------|
| Bias | Fpz | Ground |
| N8P | C3 | Left-most channel |
| N7P | C1 | Left-center channel |
| N6P | C2 | Right-center channel  |
| N5P | C4 | Right-most channel |
| N4P | HEOG | Under left eye |
| N3P | Left EMG | Left forearm |
| N2P | Right EMG | Right forearm |
| SRB | Earclip (left) | Reference |
