# Projection Mapping Software

A Python-based projection mapping software with a dual-window interface for output display and control.

## Features

- Output window for projection display
- Control window with adjustable parameters
- Test pattern generation
- Brightness control

## Requirements

- Python 3.8 or higher
- PyQt6
- OpenCV
- NumPy

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

The application will open two windows:
1. Output Window: Displays the projection output
2. Control Window: Contains controls for adjusting the projection

## Controls

- Brightness Slider: Adjust the overall brightness of the output
- Test Pattern Button: Display a test pattern with basic shapes

## Development

This is a basic implementation that can be extended with additional features such as:
- Image/video input support
- Warping and keystone correction
- Multiple projection zones
- Custom mapping patterns 