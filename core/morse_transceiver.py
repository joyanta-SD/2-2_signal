import numpy as np
import scipy.signal as signal
from config import AirBuddsConfig

MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
    'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
    'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
    'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
    'Y': '-.--', 'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.--',
    '/': '-..-.', '(': '-.--.', ')': '-.--.-', '&': '.-...', ':': '---...',
    ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-', '_': '..--.-',
    '"': '.-..-.', '$': '...-..-', '@': '.--.-.', ' ': ' '
}
REVERSE_MORSE = {v: k for k, v in MORSE_CODE_DICT.items() if k != ' '}

class MorseTransceiver:
    def __init__(self, config: AirBuddsConfig):
        self.config = config
        self.fs = config.audio.sample_rate
        # Fetch from new config structure
        self.freq = getattr(config, 'morse_freq', 800)
        self.wpm = getattr(config, 'morse_wpm', 20)
        
        # Morse timing (PARIS standard)
        # 1 WPM = 50 dots per minute -> 1 dot = 1.2 / WPM seconds
        self.dot_time = 1.2 / self.wpm
        self.dash_time = 3 * self.dot_time
        self.symbol_space = self.dot_time
        self.letter_space = 3 * self.dot_time
        self.word_space = 7 * self.dot_time

    def get_rx_constellation(self):
        return None
        
    def get_channel_estimate(self):
        return None

    def encode(self, data: bytes) -> np.ndarray:
        text = data.decode('utf-8', errors='replace').upper()
        
        # Convert to dots/dashes
        morse_chars = []
        for char in text:
            if char in MORSE_CODE_DICT:
                morse_chars.append(MORSE_CODE_DICT[char])
            elif char == ' ':
                morse_chars.append(' ')
        
        # Build audio
        audio = []
        t_dot = np.arange(int(self.dot_time * self.fs)) / self.fs
        dot_wave = np.sin(2 * np.pi * self.freq * t_dot)
        t_dash = np.arange(int(self.dash_time * self.fs)) / self.fs
        dash_wave = np.sin(2 * np.pi * self.freq * t_dash)
        
        # Apply envelope to prevent clicks (cosine taper)
        fade_len = int(0.005 * self.fs)
        if fade_len > 0:
            fade_in = np.sin(np.linspace(0, np.pi/2, fade_len))
            fade_out = np.cos(np.linspace(0, np.pi/2, fade_len))
            
            def apply_fade(wave):
                w = wave.copy()
                if len(w) > 2*fade_len:
                    w[:fade_len] *= fade_in
                    w[-fade_len:] *= fade_out
                return w
                
            dot_wave = apply_fade(dot_wave)
            dash_wave = apply_fade(dash_wave)
        
        space_sym = np.zeros(int(self.symbol_space * self.fs))
        space_let = np.zeros(int(self.letter_space * self.fs))
        # Word space is 7 dots total. If a space char is encountered, we already added 
        # a letter space (3 dots) after the previous char. We just need to add 4 more dots of silence.
        space_word = np.zeros(int(4 * self.dot_time * self.fs))
        
        # Lead in silence
        audio.append(np.zeros(int(0.5 * self.fs)))

        for c in morse_chars:
            if c == ' ':
                audio.append(space_word)
            else:
                for i, sym in enumerate(c):
                    if sym == '.':
                        audio.append(dot_wave)
                    elif sym == '-':
                        audio.append(dash_wave)
                    
                    if i < len(c) - 1:
                        audio.append(space_sym)
                # Add letter space after char
                audio.append(space_let)
                
        # Lead out silence
        audio.append(np.zeros(int(0.5 * self.fs)))

        if len(audio) == 0:
            return np.zeros(10)
            
        return np.concatenate(audio).astype(np.float32)

    def decode(self, rx_signal: np.ndarray) -> tuple[bytes, dict]:
        # Bandpass filter around tone frequency
        nyq = 0.5 * self.fs
        b, a = signal.butter(4, [(self.freq - 100) / nyq, (self.freq + 100) / nyq], btype='band')
        filtered = signal.filtfilt(b, a, rx_signal)
        
        # Envelope detection
        envelope = np.abs(signal.hilbert(filtered))
        
        # Lowpass filter the envelope
        # Cutoff frequency based on dot speed
        cutoff = min((5 / self.dot_time), nyq - 1)
        b_lp, a_lp = signal.butter(2, cutoff / nyq, btype='low')
        smooth_env = signal.filtfilt(b_lp, a_lp, envelope)
        
        # Dynamic thresholding based on K-means style or simple mean
        # Just use 40% of the maximum amplitude
        max_amp = np.max(smooth_env)
        if max_amp < 0.05: # empty or noise
            return b'', {'crc_valid': False, 'error': 'Signal too weak'}
            
        threshold = max_amp * 0.4
        binary = (smooth_env > threshold).astype(int)
        
        # Find rising/falling edges
        diff = np.diff(binary)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        if len(starts) == 0 or len(ends) == 0:
            return b'', {'crc_valid': False, 'error': 'No pulses found'}
            
        # Clean up mismatched edges
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:len(ends)]
            
        durations = (ends - starts) / self.fs
        gaps = (starts[1:] - ends[:-1]) / self.fs
        
        # Determine actual dot_time from the shortest pulses to adapt to actual speed
        # But we also have self.dot_time as a hint
        median_short = np.median([d for d in durations if d < 2 * self.dot_time])
        if not np.isnan(median_short) and median_short > 0:
            adaptive_dot_time = median_short
        else:
            adaptive_dot_time = self.dot_time

        decoded_str = ""
        current_char = ""
        
        for i in range(len(durations)):
            dur = durations[i]
            if dur > 2.0 * adaptive_dot_time:
                current_char += "-"
            elif dur > 0.3 * adaptive_dot_time: # reject very short glitches
                current_char += "."
                
            if i < len(gaps):
                gap = gaps[i]
                if gap > 4.5 * adaptive_dot_time:
                    # Word space
                    if current_char in REVERSE_MORSE:
                        decoded_str += REVERSE_MORSE[current_char]
                    else:
                        decoded_str += "?"
                    decoded_str += " "
                    current_char = ""
                elif gap > 1.8 * adaptive_dot_time:
                    # Letter space
                    if current_char in REVERSE_MORSE:
                        decoded_str += REVERSE_MORSE[current_char]
                    else:
                        decoded_str += "?"
                    current_char = ""
                    
        # Last char
        if current_char in REVERSE_MORSE:
            decoded_str += REVERSE_MORSE[current_char]
            
        # We always consider Morse code valid, CRC check is a dummy pass for the UI
        return decoded_str.encode('utf-8'), {'crc_valid': True, 'raw_durations': durations}
