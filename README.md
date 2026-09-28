# AirBudds - Acoustic Data Modem 📡🔊

**AirBudds** is a software-based acoustic data modem that allows for the transmission of digital data (like text and files) over the air using sound waves. This project was developed as our final project for the Signal Processing course. It demonstrates practical applications of various digital communication techniques, including OFDM, QAM, Morse Code, and Audio Steganography.

By relying purely on audio frequencies (via standard computer speakers and microphones), AirBudds simulates how radio frequency (RF) modems work in a highly observable and accessible environment.

---

## 🌟 Core Features

- **Multi-Mode Transmission**
  - **OFDM**: High-speed multi-carrier data transmission for files and long text.
  - **Morse Code**: A robust, low-speed legacy fallback mode.
  - **Steganography**: Hides data within a carrier music track (in the 2kHz-7kHz range) so the transmission sounds just like a regular audio file.

- **Advanced OFDM Engine**
  - Synthesizes waveforms using IFFT (Inverse Fast Fourier Transform).
  - Uses Cyclic Prefixes to combat Inter-Symbol Interference (ISI) caused by room echoes.
  - Implements Zero-Forcing channel equalization using pilot tones to correct phase drift and multipath distortion.

- **Error Correction & Integrity**
  - **Forward Error Correction (FEC)**: Uses Reed-Solomon coding to mathematically recover lost data if acoustic noise disrupts the signal.
  - **Block Interleaving**: Scrambles data bytes before transmission to protect against sudden acoustic bursts or clicks.
  - **CRC-32**: Appends checksums to ensure decoded data is 100% accurate.

- **Interactive GUI Dashboard**
  - Real-time visualizations including a Time-domain Waveform, STFT Spectrogram, and I/Q Constellation scatter plots built directly in Tkinter.

- **Seamless File Transfer**
  - Send small `.txt` or `.png` files over the acoustic channel. The receiver decodes and automatically saves the incoming file to a local `received_files/` directory.

---

## 🛠 Setup & Installation

### Requirements
- **Python**: 3.10 or higher.
- **Hardware**: A working microphone and speaker (built-in laptop audio works great).
- **Environment**: A relatively quiet room is recommended for optimal OFDM transmission, though the FEC can handle moderate background noise.

### Installation
1. Clone this repository and navigate into the project folder.
2. Install the required signal processing dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Major dependencies include NumPy, SciPy, Matplotlib, and sounddevice).*

---

## 🚀 Usage Guide

### Starting the Application
The best way to use AirBudds is through the graphical interface. 
```bash
python main.py
```
This will launch the interactive dashboard where you can toggle modes, type messages, attach files, and watch the transmission visualizations in real-time.

### Using Steganography Mode
If you switch to Steganography mode:
1. You must select a `.wav` carrier music file in the GUI. 
2. The modem will carefully encode your data beneath the music.
3. The receiver will filter the audio and extract the hidden data from the track.

---

## 📁 Project Structure

- `core/`: Contains the core digital signal processing (DSP) logic (`ofdm_transceiver.py`, `stego_transceiver.py`, `morse_transceiver.py`, `channel_estimator.py`).
- `gui/`: Houses the Tkinter application code and real-time Matplotlib visualizations.
- `protocol/`: Manages packet framing, Reed-Solomon FEC, interleaving, and bit-level conversions.
- `audio/`: Handles the direct interface with system hardware (microphone recording and speaker playback).
- `tests/`: A suite of `pytest` scripts validating the mathematical accuracy of the QAM mapping and OFDM pipeline.
- `received_files/`: This folder is generated automatically when a file is successfully received over the air.

---
*Created by Joyanta and Dipbroto for our Signal Processing Final Project.*
