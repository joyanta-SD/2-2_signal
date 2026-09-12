# 📡 AirBudds — How It Works (Simple Workflow Guide)

> **Team 8J3D9** — Joyanta Sutradhar · Dipbroto Karmokar Dip  
> Signals & Linear Systems Project

---

## 🤔 What Does AirBudds Do?

AirBudds turns **text (or files) into sound**, sends that sound through the air (via a speaker), and then **listens to the sound (via a microphone) and converts it back into text**.

Think of it like a walkie-talkie, but instead of voice, it sends *data* — and instead of radio waves, it uses **audible sound waves** that you can actually hear!

AirBudds now features a **Dual-Mode Architecture**:
1. **Morse Code Mode**: Extremely robust, classical continuous-wave transmission (perfect for very noisy rooms).
2. **OFDM Mode**: High-speed, modern broadband transmission (perfect for large payloads and clean audio).

---

## 🧩 The Big Picture (In One Sentence)

```
Your Text  →  [Encode into Sound]  →  🔊 Speaker plays it  →  🎤 Mic records it  →  [Decode back]  →  Your Text
```

That's it. Everything below is just the details of how each arrow works.

---

## 🔧 How the Encoding Works (Text → Sound)

When you press "Transmit" or "Export Audio", here's what happens step-by-step:

### Step 1: Add a Safety Check (CRC)
Before anything, the system attaches a **checksum** (CRC-32) to your message — like a receipt number. This lets the receiver verify that the data arrived correctly, without any corruption.

### Step 2: Forward Error Correction (FEC) & Interleaving
The system then adds mathematical redundancy using **Reed-Solomon Error Correction**. This allows the receiver to *fix* corrupted bytes automatically. Finally, it **interleaves** (scrambles) the data so that a sudden loud noise (like a clap) doesn't destroy an entire block of data at once.

### Step 3: Convert Text to Binary Bits
Your protected message gets converted into raw binary (0s and 1s), because computers and signals only understand numbers.

### Step 4: QAM Modulation (Bits → Points on a Map) [OFDM Mode Only]
Groups of bits are mapped to **points on a grid** called a "constellation." Each point represents a tiny chunk of data. 
- **4-QAM**: 2 bits per point (simple, very robust)
- **16-QAM**: 4 bits per point (faster, needs cleaner audio)
- **64-QAM**: 6 bits per point (fastest, needs very clean audio)

### Step 5: OFDM — Spread the Data Across Many Frequencies [OFDM Mode Only]
Instead of putting all data on one single tone (like Morse Code does), OFDM spreads it across **hundreds of tiny frequency channels** simultaneously — like playing a chord on a piano instead of a single key. This makes the signal extremely high-speed and robust against frequency-specific echoes.

Each OFDM symbol also has:
- **Pilot tones**: Known reference signals (like landmarks) that help the receiver figure out what the room's echo did to the sound.
- **Cyclic Prefix**: A short copy of the end of the symbol, tacked onto the beginning, so echoes don't smash one symbol into the next.

### Step 6: Add a Sync Chirp (Preamble)
A quick **dual chirp** (a sound that sweeps from low pitch to high pitch, twice) is placed at the very beginning. This chirp acts as a "start marker" — when the receiver hears it, it knows exactly where the data begins.

### Step 7: Play the Sound! 🔊
The final audio waveform is played through the speaker (or saved to a `.wav` file). What you hear is a short chirp followed by a burst of buzzy noise (in OFDM) or beeps (in Morse) — that is actually your data!

---

## 👂 How the Decoding Works (Sound → Text)

When you press "Start Listening" or "Upload & Decode", here's the reverse process:

### Step 1: Find the Start & Correct Clock Drift (Schmidl-Cox Sync)
The receiver scans the recorded audio for the **chirp preamble**. Using **Schmidl-Cox Synchronization** on the repeated chirps, it not only finds where the data starts, but also calculates exactly how much the sender's audio clock is drifting compared to the receiver's clock (Carrier Frequency Offset).

### Step 2: Automatic Gain Control (AGC)
Different speakers and microphones have different volume levels. AGC automatically adjusts the volume of the received signal so it's always at the right level for decoding — no manual volume twiddling needed.

### Step 3: Chop into OFDM Symbols
The receiver slices the audio into equal-sized chunks, strips the cyclic prefix from each one, and converts them back to the frequency domain using the **FFT** (Fast Fourier Transform). Now we can see what data was on each frequency channel.

### Step 4: Channel Estimation & Equalization
The room's acoustics (echoes, frequency response of the speaker/mic) will have distorted the signal. The receiver uses the **pilot tones** to figure out exactly how the channel distorted each frequency, and then mathematically reverses that distortion. This is called **Zero-Forcing Equalization**.

### Step 5: QAM Demodulation (Points → Bits) [OFDM Mode Only]
Each received point on the constellation is snapped to the nearest known grid point, recovering the original bits.

### Step 6: Deinterleave & FEC Decode
The bits are reconstructed into bytes, unscrambled back into their original order, and fed into the **Reed-Solomon Decoder**. If the room echoes caused any minor errors, this mathematical safety net will detect and fix them!

### Step 7: CRC Verification
Finally, the recovered message's CRC checksum is verified. If it matches → ✅ the data is perfect. If it doesn't → ⚠️ something went wrong (too much noise, bad microphone, etc.).

---

## 🖥️ Three Ways to Use AirBudds

### Mode 1: Live Speaker → Microphone (Real-Time)

> **Best for:** Demos, showing the system works live.

1. Open the GUI: `python main.py`
2. Select your Mode: **Morse Code** or **OFDM**.
3. Type your message in the **"TX Message"** box.
4. On **Computer A** (sender): Click **"Transmit Text (Speaker)"**.
5. On **Computer B** (receiver): Click **"Start Listening"** *before* the sender transmits.
6. The receiver's microphone picks up the sound, decodes it, and shows the result in **"Decoded RX Result"**.

> ⚠️ **Tips for live mode:**
> - Keep the sender's speaker volume at 80–100%.
> - Place the microphone within 1 meter of the speaker.
> - Reduce background noise (fans, music, talking).
> - Use "Test Mic Level" to verify your microphone is working.

---

### Mode 2: Export to WAV File (Most Reliable ✅)

> **Best for:** Reliable transfers, sending data between devices, sharing with friends.

1. Open the GUI: `python main.py`
2. Type your message in the **"TX Message"** box.
3. Click **"💾 Export Audio (.wav)"**.
4. Choose where to save the file.
5. A `.wav` file is created containing your encoded message as sound.
6. You can now:
   - Play it on **any device** (phone, tablet, another computer).
   - Share it over WhatsApp, email, USB drive, etc.
   - The receiver just needs to upload it back to AirBudds (Mode 3).

---

### Mode 3: Upload & Decode a WAV File

> **Best for:** Decoding audio received from Mode 2, or recordings from external devices.

1. Open the GUI: `python main.py`
2. Click **"📂 Upload & Decode (.wav)"**.
3. Select the `.wav` file you want to decode.
4. AirBudds will automatically:
   - Convert the sample rate if needed (e.g., phone recordings at 48 kHz → AirBudds' 44.1 kHz).
   - Convert stereo to mono if needed.
   - Decode the message and show it in **"Decoded RX Result"**.
5. A popup will tell you whether the CRC check passed (✅) or failed (⚠️).

---

## 🧪 Offline Simulation (Testing Without Hardware)

You don't need a speaker or microphone to test the system!

1. Open the GUI: `python main.py`
2. Type a message.
3. Choose a **Room Preset** (small_room, hallway, open_air, anechoic).
4. Set the **SNR** (Signal-to-Noise Ratio) — higher = cleaner signal.
5. Click **"Run Simulation & Equalize"**.
6. The system will:
   - Encode your message.
   - Simulate a real room (echoes + noise) mathematically.
   - Decode the distorted signal.
   - Show you the results — including the constellation plot showing how well equalization cleaned up the data.

---

## 📊 Understanding the Dashboard Plots

| Plot | What It Shows |
|------|---------------|
| **Waveform** | The raw audio signal in the time domain — you'll see the chirp followed by OFDM data |
| **Spectrogram** | A frequency-vs-time heatmap — you'll see the chirp sweeping up, then the OFDM data as a band of energy across multiple frequencies |
| **Constellation** | The QAM grid — after equalization, points should cluster tightly around the ideal grid locations. Scattered points = high noise |

---

## 🔁 Summary Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRANSMIT SIDE                               │
│                                                                     │
│   "Hello!"  →  CRC  →  Reed-Solomon FEC  →  Bits  →  QAM Map        │
│       →  OFDM (IFFT + CP)  →  Chirp Prepend  →  🔊 Speaker         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓  🔈  sound waves  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         RECEIVE SIDE                                │
│                                                                     │
│   🎤 Mic / 📂 WAV Upload  →  Schmidl-Cox Sync  →  AGC  →  FFT       │
│       →  Pilot Channel Est.  →  Equalize  →  QAM Demap            │
│       →  FEC Decode & Fix Errors  →  CRC  →  "Hello!"  ✅           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 💡 Quick FAQ

**Q: Why does the transmitted audio sound like buzzy noise?**  
A: That's your data! Hundreds of sine waves at different frequencies playing simultaneously. It sounds like noise to human ears, but to AirBudds, each frequency carries a piece of your message.

**Q: Why does the system add a chirp at the beginning?**  
A: The chirp is a known pattern. The receiver uses it to find exactly where in the recording the data starts — precise to the sample level.

**Q: What is a Cyclic Prefix?**  
A: When sound bounces off walls (echoes), the end of one OFDM symbol can bleed into the beginning of the next one. The cyclic prefix is a "guard interval" that absorbs these echoes, keeping the data clean.

**Q: Why is WAV export more reliable than live speaker-to-mic?**  
A: Live audio goes through the speaker → air → microphone chain, which adds room echo, background noise, and hardware distortion. WAV export skips all of that — the encoded signal is digitally perfect.

**Q: What does CRC FAIL mean?**  
A: It means the received data doesn't match the original. This happens when there's too much noise or the signal was distorted beyond recovery. Try reducing distance, increasing volume, or using WAV export mode.

---

*AirBudds — Turning sound into data, one subcarrier at a time.* 🎵📡
