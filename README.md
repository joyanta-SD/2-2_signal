# 🔊 AirBudds — Acoustic OFDM Data Transceiver & Signal Analyzer

> **Course:** Signals & Linear Systems  
> **Team:** 8J3D9  
> **Members:** Joyanta Sutradhar (Roll: 2305083) · Dipbroto Karmokar Dip (Roll: 2305089)

---

## 📖 Project Description

AirBudds is an interactive software application that **transmits and receives digital data (text or small files) over the air using sound waves**. By implementing Orthogonal Frequency Division Multiplexing (OFDM) and Quadrature Amplitude Modulation (QAM), AirBudds transforms a standard computer speaker and microphone into a complete digital communication transceiver.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        AirBudds System                          │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│  main.py │ config.py│  core/   │   dsp/   │ protocol/│   gui/   │
│ (Entry)  │ (Config) │ (Engine) │ (Utils)  │ (Link)   │(Dashboard│
├──────────┴──────────┴──────────┴──────────┴──────────┴──────────┤
│                          audio/                                  │
│                   (Speaker & Microphone I/O)                     │
└──────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

| Feature | Module | Description |
|---------|--------|-------------|
| **OFDM Modulation** | `core/ofdm_transceiver.py` | IFFT synthesis, cyclic prefix, multi-carrier transmission |
| **QAM Mapping** | `core/qam_mapper.py` | 4/16/64-QAM with Gray coding |
| **Preamble Sync** | `core/preamble.py` | LFM chirp matched filtering for microsecond timing |
| **Channel Estimation** | `core/channel_estimator.py` | Pilot-tone estimation + Zero-Forcing/MMSE equalization |
| **LMS Noise Cancel** | `dsp/lms_filter.py` | Adaptive filter to subtract steady background noise |
| **RIR Simulator** | `dsp/rir_simulator.py` | Offline room echo simulation via FIR convolution |
| **Ultrasonic Mode** | `dsp/spectral_shift.py` | Silent 18–20 kHz transmission mode |
| **Audio Watermark** | `dsp/watermark.py` | Spread-spectrum sync tone detection |
| **CRC Integrity** | `core/crc.py` | CRC-32 checksum for error detection |
| **Auto Retry** | `protocol/retry_manager.py` | ARQ retransmission on CRC failure |
| **Live Dashboard** | `gui/dashboard.py` | Real-time waveform, spectrogram, and constellation plots |

## 📦 Installation

```bash
# Clone or navigate to the project directory
cd "final signal project"

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.10
- NumPy ≥ 1.24
- SciPy ≥ 1.11
- Matplotlib ≥ 3.7
- sounddevice ≥ 0.4
- PyAudio ≥ 0.2 (optional fallback)
- pytest ≥ 7.0 (for testing)

## 🚀 Usage

### GUI Mode (Default)

```bash
python main.py
```

Launches the interactive dashboard with waveform, spectrogram, and constellation plots.

### Transmit Text

```bash
python main.py --tx "Hello AirBudds!"
python main.py --tx "Secret message" --ultrasonic    # Silent mode
python main.py --tx "Hi" --qam 16                     # 16-QAM
```

### Receive & Decode

```bash
python main.py --rx --duration 5
python main.py --rx --duration 10 --noise-cancel      # With LMS filter
```

### Offline Simulation

```bash
python main.py --sim --preset small_room --snr 25
python main.py --sim --preset hallway --snr 15 --qam 16 --verbose
```

### All Options

```
usage: AirBudds [-h] [--gui | --tx MSG | --tx-file PATH | --rx | --sim]
                [--duration SEC] [--qam {4,16,64}] [--ultrasonic]
                [--preset {anechoic,small_room,hallway,open_air}]
                [--snr DB] [--noise-cancel] [--watermark] [--retries N] [-v]
```

## 🔬 Signal Processing Pipeline

### Transmitter (TX)

```
Data → UTF-8 → CRC-32 Append → Bits → QAM Modulation → Pilot Insertion
    → IFFT → Cyclic Prefix → Preamble Prepend → [Optional: Ultrasonic Shift]
    → [Optional: Watermark Embed] → Speaker Playback
```

### Receiver (RX)

```
Microphone → [Optional: Watermark Detect] → [Optional: Ultrasonic Downshift]
    → [Optional: LMS Noise Cancel] → Preamble Matched Filter → Frame Sync
    → CP Removal → FFT → Pilot Extraction → Channel Estimation
    → Zero-Forcing/MMSE Equalization → QAM Demodulation → CRC Verify → Data
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_ofdm.py -v
python -m pytest tests/test_qam.py -v
```

## 📁 Project Structure

```
final signal project/
├── main.py                     # Application entry point
├── config.py                   # System parameters (dataclasses)
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── core/                       # Core DSP Engine
│   ├── ofdm_transceiver.py     # Full OFDM TX/RX pipeline
│   ├── qam_mapper.py           # QAM constellation mapping
│   ├── preamble.py             # LFM chirp sync
│   ├── channel_estimator.py    # Pilot-tone channel estimation
│   ├── synchronizer.py         # Timing & CFO correction
│   └── crc.py                  # CRC-32 checksum
│
├── audio/                      # Audio Hardware Interface
│   ├── audio_interface.py      # sounddevice wrapper
│   ├── recorder.py             # Mic recording buffer
│   └── player.py               # Speaker playback buffer
│
├── dsp/                        # Signal Processing Utilities
│   ├── filters.py              # FIR/IIR digital filters
│   ├── lms_filter.py           # LMS adaptive noise cancellation
│   ├── rir_simulator.py        # Room impulse response simulator
│   ├── spectral_shift.py       # Ultrasonic frequency shifting
│   └── watermark.py            # Spread-spectrum watermark
│
├── protocol/                   # Link-Layer Protocol
│   ├── frame_builder.py        # Packet framing
│   ├── retry_manager.py        # ARQ retransmission
│   └── codec.py                # Binary ↔ text/file codec
│
├── gui/                        # Interactive Dashboard
│   ├── dashboard.py            # Main Tkinter window
│   ├── waveform_plot.py        # Time-domain plot
│   ├── spectrogram_plot.py     # STFT spectrogram
│   ├── constellation_plot.py   # I/Q constellation scatter
│   └── controls.py             # UI controls & sliders
│
└── tests/                      # Test Suite
    ├── test_ofdm.py            # OFDM round-trip tests
    ├── test_qam.py             # QAM accuracy tests
    ├── test_crc.py             # CRC verification tests
    ├── test_preamble.py        # Chirp sync tests
    ├── test_channel.py         # Channel estimation tests
    ├── test_rir.py             # RIR simulation tests
    └── test_lms.py             # LMS convergence tests
```

## 📐 Mathematical Foundations

### OFDM Symbol (IFFT Synthesis)

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{j 2\pi k n / N}, \quad n = 0, 1, \ldots, N-1$$

### Cyclic Prefix

$$x_{cp}[n] = [x[N - L_{cp}], \ldots, x[N-1], x[0], \ldots, x[N-1]]$$

### LFM Chirp Preamble

$$s(t) = w(t) \cdot \cos\!\left(2\pi\left(f_0 t + \frac{B}{2T} t^2\right)\right)$$

### Zero-Forcing Equalization

$$\hat{X}[k] = \frac{Y[k]}{H[k]}$$

### MMSE Equalization

$$\hat{X}[k] = \frac{H^*[k]}{|H[k]|^2 + \sigma_n^2 / \sigma_x^2} \cdot Y[k]$$

### LMS Adaptive Filter

$$\mathbf{w}[n+1] = \mathbf{w}[n] + 2\mu \cdot e[n] \cdot \mathbf{x}[n]$$

### CRC-32 Polynomial

$$G(x) = x^{32} + x^{26} + x^{23} + x^{22} + x^{16} + x^{12} + x^{11} + x^{10} + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1$$

---

*AirBudds — Turning sound into data, one subcarrier at a time.* 🎵📡
