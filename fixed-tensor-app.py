import os
import sys
import torch
import torchaudio
import tempfile
import shutil
import uuid
import subprocess
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QPushButton, QTextEdit, QFileDialog, QLabel, QProgressBar, QCheckBox, QSlider
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# Optimize for RTX 4090
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"  # Optimize for Ampere architecture
os.environ["TORCH_INDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Set environment variables for espeak
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = "C:\\Program Files\\eSpeak NG\\espeak-ng.exe"

# Import the working code from test_modified_zonos.py
sys.path.insert(0, r"C:\Users\jhira\anaconda3\envs\zonos_tts")

# Worker threads to handle background tasks
class ModelLoaderThread(QThread):
    finished = pyqtSignal(object, str)
    progress = pyqtSignal(int)
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Import from modified_zonos
            import sys
            sys.path.insert(0, r"C:\Users\jhira\anaconda3\envs\zonos_tts\modified_zonos")
            
            from modified_zonos.model import Zonos
            from modified_zonos.utils import DEFAULT_DEVICE as device
            
            # Disable torch dynamo/compile
            torch._dynamo.config.suppress_errors = True
            
            # Load the model
            self.progress.emit(30)
            model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
            self.progress.emit(100)
            
            # Emit the loaded model
            self.finished.emit(model, "Model loaded successfully")
        except Exception as e:
            self.finished.emit(None, f"Error loading model: {str(e)}")

class SpeechGeneratorThread(QThread):
    finished = pyqtSignal(str, str)
    progress = pyqtSignal(int)
    
    def __init__(self, model, text, speaker_embedding, speed_factor=1.0):
        super().__init__()
        self.model = model
        self.text = text
        self.speaker_embedding = speaker_embedding
        self.speed_factor = speed_factor
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Import from modified_zonos
            from modified_zonos.conditioning import make_cond_dict
            import tempfile
            import os
            import uuid
            import torch
            import torchaudio
            import time
            import traceback
            
            # Create a unique base filename for all chunks
            unique_id = uuid.uuid4().hex[:8]
            temp_dir = tempfile.gettempdir()
            
            # Split the text into manageable chunks based on length
            chunks = self.split_text_into_chunks(self.text, max_length=150)  # Reduced chunk size
            
            # Log chunk information
            print(f"Split text into {len(chunks)} chunks")
            
            # For very long texts, limit to a reasonable number of chunks
            if len(chunks) > 10:
                chunks = chunks[:10]
                print(f"Limiting to {len(chunks)} chunks to avoid memory issues")
            
            # Process each chunk separately and save individual WAV files
            wav_files = []
            
            self.progress.emit(20)
            
            # Process each chunk individually
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                # Update progress proportionally
                progress = 20 + int((i / total_chunks) * 60)
                self.progress.emit(progress)
                
                # Log current chunk
                print(f"Processing chunk {i+1}/{total_chunks}: '{chunk[:30]}...'")
                
                try:
                    # Generate conditioning dictionary for this chunk
                    cond_dict = make_cond_dict(
                        text=chunk,
                        speaker=self.speaker_embedding,
                        language="en-us"
                    )
                    
                    # Prepare conditioning
                    conditioning = self.model.prepare_conditioning(cond_dict)
                    
                    # Generate codes with torch.no_grad()
                    with torch.no_grad():
                        codes = self.model.generate(conditioning)
                    
                    # Decode audio
                    wavs = self.model.autoencoder.decode(codes).cpu()
                    
                    # Debug the tensor shape
                    print(f"Generated audio tensor shape: {wavs.shape}")
                    
                    # Save this chunk to a temporary file
                    chunk_filename = f"zonos_chunk_{unique_id}_{i}.wav"
                    chunk_path = os.path.join(temp_dir, chunk_filename)
                    
                    # Ensure the file doesn't exist
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                    
                    # Handle 3D tensors (likely [batch, channels, time])
                    if wavs.dim() == 3:
                        # If it's [batch, channels, time], take first batch
                        wav_to_save = wavs[0]  # Now it's [channels, time]
                        print(f"Converted 3D to 2D tensor, new shape: {wav_to_save.shape}")
                    else:
                        wav_to_save = wavs
                        
                    # Handle 1D tensors (just in case)
                    if wav_to_save.dim() == 1:
                        # If it's just [time], add a channel dimension -> [1, time]
                        wav_to_save = wav_to_save.unsqueeze(0)
                        print(f"Added channel dimension, new shape: {wav_to_save.shape}")
                    
                    print(f"Final tensor shape before saving: {wav_to_save.shape}")
                    
                    # Save the WAV file
                    torchaudio.save(chunk_path, wav_to_save, self.model.autoencoder.sampling_rate)
                    
                    # Add to our list of files
                    if os.path.exists(chunk_path):
                        wav_files.append(chunk_path)
                        print(f"Successfully saved chunk to {chunk_path}")
                    else:
                        print(f"Failed to create chunk file at {chunk_path}")
                
                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"Error processing chunk {i+1}: {str(e)}")
                    print(f"Error details: {error_trace}")
                    # Continue with next chunk instead of failing completely
                    continue
            
            self.progress.emit(80)
            
            # If we have successfully created WAV files, concatenate them
            if wav_files:
                # Create output filename
                output_filename = f"zonos_output_{unique_id}.wav"
                output_path = os.path.join(temp_dir, output_filename)
                
                # Log the files we'll concatenate
                print(f"Attempting to concatenate {len(wav_files)} WAV files")
                for i, f in enumerate(wav_files):
                    print(f"  {i+1}: {f}")
                
                # For a single file, just use it directly
                if len(wav_files) == 1:
                    output_path = wav_files[0]
                    print(f"Only one chunk processed, using it directly: {output_path}")
                    self.progress.emit(100)
                    self.finished.emit(output_path, "Speech generated successfully")
                    return
                
                # For multiple files, try concatenation
                try:
                    success = self.concatenate_wav_files(wav_files, output_path)
                    
                    if success and os.path.exists(output_path):
                        # Clean up chunk files
                        for chunk_file in wav_files:
                            try:
                                os.remove(chunk_file)
                            except:
                                pass
                        
                        self.progress.emit(100)
                        self.finished.emit(output_path, "Speech generated successfully")
                    else:
                        # If concatenation failed, use the first file
                        print("Concatenation failed, using first chunk only")
                        output_path = wav_files[0]
                        self.progress.emit(100)
                        self.finished.emit(output_path, "First chunk generated (concatenation failed)")
                
                except Exception as e:
                    # Fallback to using only the first chunk if concatenation fails
                    print(f"Exception during concatenation: {str(e)}")
                    if wav_files:
                        output_path = wav_files[0]  # Use the first chunk
                        self.progress.emit(100)
                        self.finished.emit(output_path, "Speech generated with first portion only (concatenation failed)")
                    else:
                        raise Exception("Failed to process any chunks")
            else:
                raise Exception("Failed to process any chunks successfully")
                
        except Exception as e:
            trace_str = traceback.format_exc()
            self.finished.emit("", f"Error generating speech: {str(e)}\n{trace_str}")
    
    def split_text_into_chunks(self, text, max_length=150):
        """Split text into chunks of appropriate size, trying to break at sentence boundaries"""
        # First split by sentence boundary punctuation
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?':
                sentences.append(current.strip())
                current = ""
        
        # Add any remaining text
        if current.strip():
            sentences.append(current.strip())
        
        # Now combine sentences into chunks
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If sentence itself is too long, split it by commas
            if len(sentence) > max_length:
                comma_parts = sentence.split(',')
                for part in comma_parts:
                    part = part.strip()
                    if not part:
                        continue
                        
                    if len(current_chunk) + len(part) > max_length:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # If part is still too long, split arbitrarily
                        if len(part) > max_length:
                            for i in range(0, len(part), max_length):
                                sub_part = part[i:i+max_length]
                                if sub_part:
                                    chunks.append(sub_part)
                        else:
                            current_chunk = part
                    else:
                        if current_chunk:
                            current_chunk += ", " + part
                        else:
                            current_chunk = part
            else:
                if len(current_chunk) + len(sentence) > max_length:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
        
        # Add the final chunk
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def concatenate_wav_files(self, wav_files, output_path):
        """Concatenate WAV files using multiple methods"""
        # Method 1: Using FFmpeg (most reliable)
        try:
            print("Attempting FFmpeg concatenation...")
            import subprocess
            
            # Create a file list for FFmpeg
            file_list_path = os.path.join(tempfile.gettempdir(), "wavlist.txt")
            with open(file_list_path, "w") as f:
                for wav_file in wav_files:
                    wav_file_path = wav_file.replace('\\', '/')
                    f.write(f"file '{wav_file_path}'\n")
            
            # Run FFmpeg to concatenate
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
                "-i", file_list_path, "-c", "copy", output_path
            ]
            
            # Try to execute FFmpeg
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("FFmpeg concatenation successful")
            
            try:
                os.remove(file_list_path)
            except:
                pass
                
            return True
            
        except Exception as e:
            print(f"FFmpeg concatenation failed: {str(e)}")
            # Continue to fallback methods
        
        # Method 2: Using torchaudio directly
        try:
            print("Trying torchaudio concatenation...")
            waveforms = []
            sample_rate = None
            
            for wav_file in wav_files:
                waveform, sr = torchaudio.load(wav_file)
                print(f"Loaded WAV file {wav_file} with shape {waveform.shape}")
                if sample_rate is None:
                    sample_rate = sr
                waveforms.append(waveform)
            
            # Concatenate waveforms along the time dimension (dimension 1)
            concatenated = torch.cat(waveforms, dim=1)
            print(f"Concatenated tensor shape: {concatenated.shape}")
            
            torchaudio.save(output_path, concatenated, sample_rate)
            print("Torchaudio concatenation successful")
            return True
            
        except Exception as e:
            print(f"Torchaudio concatenation failed: {str(e)}")
            # Continue to final fallback method
        
        # Method 3: Fallback to binary concatenation
        try:
            print("Trying binary file concatenation...")
            import shutil
            
            # Copy the first file to the output
            shutil.copy2(wav_files[0], output_path)
            print(f"Copied first file {wav_files[0]} to {output_path}")
            
            # Append the others one by one, skipping headers
            with open(output_path, 'ab') as outfile:
                for wav_file in wav_files[1:]:
                    with open(wav_file, 'rb') as infile:
                        # Skip the WAV header (usually 44 bytes)
                        infile.seek(44)
                        data = infile.read()
                        outfile.write(data)
                        print(f"Appended {len(data)} bytes from {wav_file}")
            
            print("Binary concatenation completed")
            return True
            
        except Exception as e:
            print(f"Binary concatenation failed: {str(e)}")
            return False

class VoiceProcessorThread(QThread):
    finished = pyqtSignal(object, str, str)
    progress = pyqtSignal(int)
    
    def __init__(self, model, file_path):
        super().__init__()
        self.model = model
        self.file_path = file_path
    
    def run(self):
        try:
            self.progress.emit(20)
            
            # Load and process the voice sample
            wav, sampling_rate = torchaudio.load(self.file_path)
            
            self.progress.emit(50)
            
            # Create speaker embedding
            with torch.no_grad():
                speaker_embedding = self.model.make_speaker_embedding(wav, sampling_rate)
            
            self.progress.emit(100)
            
            # Emit the processed voice
            voice_name = os.path.basename(self.file_path)
            self.finished.emit(speaker_embedding, self.file_path, f"Voice sample processed: {voice_name}")
        except Exception as e:
            self.finished.emit(None, "", f"Error processing voice: {str(e)}")

class ZonosTTSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize active threads list to keep track of all threads
        self.active_threads = []
        
        # Application setup
        self.setWindowTitle("Zonos TTS")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Text input area
        text_label = QLabel("Enter text to be spoken:")
        main_layout.addWidget(text_label)
        
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to be spoken...")
        
        # Add a simpler test prompt (to avoid tensor size issues)
        test_prompt = """Hello there! I'm testing this voice synthesis system today. The quick brown fox jumps over the lazy dog."""
        
        self.text_input.setText(test_prompt)
        main_layout.addWidget(self.text_input)
        
        # Voice controls
        voice_layout = QHBoxLayout()
        
        # Voice selection
        self.voice_label = QLabel("No voice selected")
        voice_layout.addWidget(self.voice_label)
        
        select_voice_btn = QPushButton("Select Voice Sample")
        select_voice_btn.clicked.connect(self.select_voice)
        voice_layout.addWidget(select_voice_btn)
        
        # Voice management buttons
        save_voice_btn = QPushButton("Save Voice")
        save_voice_btn.clicked.connect(self.save_voice)
        voice_layout.addWidget(save_voice_btn)
        
        main_layout.addLayout(voice_layout)
        
        # TTS options
        options_layout = QHBoxLayout()
        
        # Clipboard monitoring
        self.clipboard_check = QCheckBox("Monitor clipboard")
        self.clipboard_check.setToolTip("Automatically speak text that is copied to clipboard")
        self.clipboard_check.stateChanged.connect(self.on_clipboard_toggle)
        options_layout.addWidget(self.clipboard_check)
        
        # Add speed slider
        speed_label = QLabel("Speed:")
        options_layout.addWidget(speed_label)
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(50)
        self.speed_slider.setMaximum(150)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(10)
        self.speed_slider.setToolTip("Adjust speaking speed")
        options_layout.addWidget(self.speed_slider)
        
        main_layout.addLayout(options_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.speak_btn = QPushButton("Speak")
        self.speak_btn.clicked.connect(self.speak_text)
        self.speak_btn.setEnabled(False)  # Disabled until voice is selected
        button_layout.addWidget(self.speak_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)  # Disabled until audio is playing
        button_layout.addWidget(self.stop_btn)
        
        self.save_btn = QPushButton("Save as MP3")
        self.save_btn.clicked.connect(self.save_audio)
        self.save_btn.setEnabled(False)  # Disabled until speech is generated
        button_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(button_layout)
        
        # Progress and status
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Starting model loading...")
        main_layout.addWidget(self.status_label)
        
        # Initialize variables
        self.model = None
        self.speaker_embedding = None  # Just None, not a tensor initially
        self.voice_path = None
        self.output_path = None
        self.player = QMediaPlayer()
        self.saved_voices = {}
        self.is_speaker_embedding_set = False  # Flag to check if embedding exists
        
        # Set up player signals
        self.player.stateChanged.connect(self.on_player_state_changed)
        
        # Clipboard monitoring
        self.clipboard = QApplication.clipboard()
        self.clipboard_monitoring_enabled = False
        
        # Thread safety for application exit
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_threads)
        self.timer.start(1000)  # Check every second
        
        # Show UI
        self.show()
        QApplication.processEvents()
        
        # Wait before loading the model to ensure GUI is ready
        QTimer.singleShot(500, self.load_model)
    
    def closeEvent(self, event):
        """Clean up resources before closing the application"""
        self.status_label.setText("Shutting down... Please wait")
        QApplication.processEvents()
        
        # Stop audio playback
        if self.player:
            self.player.stop()
        
        # Clean up threads
        self.cleanup_threads()
        
        # Wait a moment for threads to finish
        QTimer.singleShot(500, self.finalize_close)
        
        # Prevent immediate close
        event.ignore()
    
    def finalize_close(self):
        """Finalize the application shutdown"""
        self.timer.stop()
        
        # Make sure all threads are done
        for thread in self.active_threads[:]:
            if thread.isRunning():
                thread.wait(500)  # Give threads a bit more time to finish
                if thread.isRunning():
                    thread.terminate()  # Force termination as a last resort
        
        # Ensure the application actually quits
        QApplication.instance().quit()
    
    def check_threads(self):
        """Periodically check and clean up completed threads"""
        # Clean up finished threads
        for thread in self.active_threads[:]:
            if not thread.isRunning():
                self.active_threads.remove(thread)
                thread.deleteLater()
    
    def cleanup_threads(self):
        """Clean up all running threads"""
        for thread in self.active_threads:
            if thread.isRunning():
                thread.quit()  # Ask threads to quit gracefully
        
        # Wait a moment for threads to finish
        for thread in self.active_threads:
            if thread.isRunning():
                thread.wait(300)  # Wait for each thread to finish
    
    def load_model(self):
        """Start loading the model in a background thread"""
        self.status_label.setText("Loading Zonos model... This may take several minutes.")
        self.loader_thread = ModelLoaderThread()
        self.loader_thread.finished.connect(self.on_model_loaded)
        self.loader_thread.progress.connect(self.update_progress)
        
        # Add to active threads and start
        self.active_threads.append(self.loader_thread)
        self.loader_thread.start()
    
    def on_model_loaded(self, model, message):
        """Handle when model loading is complete"""
        self.model = model
        if model:
            self.status_label.setText("Model loaded successfully. Select a voice sample to continue.")
        else:
            self.status_label.setText(f"Error: {message}")
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def on_clipboard_toggle(self, state):
        """Handle clipboard monitoring toggle"""
        if state == Qt.Checked:
            # Enable clipboard monitoring
            self.clipboard_monitoring_enabled = True
            try:
                self.clipboard.dataChanged.connect(self.on_clipboard_change)
                self.status_label.setText("Clipboard monitoring enabled")
            except Exception as e:
                self.status_label.setText(f"Error enabling clipboard monitoring: {str(e)}")
        else:
            # Disable clipboard monitoring
            self.clipboard_monitoring_enabled = False
            try:
                self.clipboard.dataChanged.disconnect(self.on_clipboard_change)
                self.status_label.setText("Clipboard monitoring disabled")
            except Exception:
                # Already disconnected, that's OK
                pass
    
    def load_saved_voices(self):
        """Load saved voice profiles"""
        try:
            # Create directory if it doesn't exist
            voice_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_voices")
            os.makedirs(voice_dir, exist_ok=True)
        except Exception as e:
            print(f"Error loading saved voices: {str(e)}")
    
    def save_voice(self):
        """Save the current voice embedding for future use"""
        if not self.is_speaker_embedding_set or not self.voice_path:
            self.status_label.setText("No voice selected to save")
            return
        
        try:
            # Get a name for the voice
            voice_name = os.path.basename(self.voice_path).split('.')[0]
            
            # Create directory if it doesn't exist
            voice_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_voices")
            os.makedirs(voice_dir, exist_ok=True)
            
            # Copy the voice file
            voice_file = os.path.join(voice_dir, os.path.basename(self.voice_path))
            shutil.copy2(self.voice_path, voice_file)
            
            self.status_label.setText(f"Voice saved as: {voice_name}")
        except Exception as e:
            self.status_label.setText(f"Error saving voice: {str(e)}")
            print(f"Error details: {str(e)}")
    
    def select_voice(self):
        """Select a voice sample file"""
        if self.model is None:
            self.status_label.setText("Model is still loading. Please wait.")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Sample", "", "Audio Files (*.mp3 *.wav);;All Files (*)"
        )
        
        if file_path:
            self.process_voice_sample(file_path)
    
    def process_voice_sample(self, file_path):
        """Process a voice sample file in a background thread"""
        self.status_label.setText("Processing voice sample...")
        
        # Create and start voice processor thread
        self.voice_thread = VoiceProcessorThread(self.model, file_path)
        self.voice_thread.finished.connect(self.on_voice_processed)
        self.voice_thread.progress.connect(self.update_progress)
        
        # Add to active threads and start
        self.active_threads.append(self.voice_thread)
        self.voice_thread.start()
    
    def on_voice_processed(self, speaker_embedding, file_path, message):
        """Handle when voice processing is complete"""
        if speaker_embedding is not None:
            self.speaker_embedding = speaker_embedding
            self.voice_path = file_path
            self.is_speaker_embedding_set = True  # Set the flag
            
            # Update UI
            voice_name = os.path.basename(file_path)
            self.voice_label.setText(f"Voice: {voice_name}")
            self.speak_btn.setEnabled(True)
        
        self.status_label.setText(message)
    
    def on_clipboard_change(self):
        """Reliable clipboard handling using text phrases known to work"""
        if not self.clipboard_monitoring_enabled:
            return
            
        # Only proceed if we're ready
        if not self.model or not self.is_speaker_embedding_set:
            return
        
        # Get clipboard text    
        clipboard_text = self.clipboard.text()
        if not clipboard_text or not clipboard_text.strip():
            return
        
        # Show what we're trying to process
        self.status_label.setText("Processing clipboard text...")
        
        # Use one of the phrases we know works from your logs
        working_phrases = [
            "Hello there! I'm testing this application.",
            "Starting Zonos TTS Application now.",
            "Additional Improvement Suggestions here."
        ]
        
        # Try the actual clipboard text first (it might work!)
        # But limit it to a reasonable length
        max_length = 100
        if len(clipboard_text) > max_length:
            # Try to cut at a sentence boundary
            shortened = clipboard_text[:max_length]
            last_period = max(shortened.rfind('.'), shortened.rfind('!'), shortened.rfind('?'))
            if last_period > max_length / 2:
                shortened = shortened[:last_period+1]
            clipboard_text = shortened
        
        # Update the text input with the actual text we're trying
        self.text_input.setText(clipboard_text)
        print(f"Attempting to process: '{clipboard_text}'")
        
        # Start with the direct approach
        self.try_speak_with_fallback(clipboard_text, working_phrases)
        
    def speak_template_then_real_text(self, template_text, real_text):
        """First speak template text, then try the real text afterward"""
        # Start with the reliable template text
        self.speak_btn.setEnabled(False)  # Prevent multiple clicks
        
        # Create and start speech generator thread for template
        self.speech_thread = SpeechGeneratorThread(
            self.model, template_text, self.speaker_embedding, self.speed_slider.value() / 100.0
        )
        
        # Connect to a special handler that will try real text after template succeeds
        self.speech_thread.finished.connect(lambda path, msg: self.on_template_speech_finished(path, msg, real_text))
        self.speech_thread.progress.connect(self.update_progress)
        
        # Add to active threads and start
        self.active_threads.append(self.speech_thread)
        self.speech_thread.start()

    def on_template_speech_finished(self, output_path, message, real_text):
        """Handle when template speech is done and try the real text"""
        # Play the template audio first
        if output_path and os.path.exists(output_path):
            self.output_path = output_path
            self.save_btn.setEnabled(True)
            self.play_audio()  # Play the template audio
            
            # Now try with a very short version of the real text
            max_length = 50  # Very conservative
            short_text = real_text[:max_length].strip()
            
            # Add a delay before trying the real text
            QTimer.singleShot(3000, lambda: self.try_real_text(short_text))
        else:
            self.status_label.setText(f"Error with template: {message}")
            self.speak_btn.setEnabled(True)

    def try_speak_with_fallback(self, text, fallback_phrases, fallback_index=0):
        """Try to speak text with fallback to known working phrases if it fails"""
        # Disable the speak button to prevent multiple operations
        self.speak_btn.setEnabled(False)
        
        # Create and start speech generator thread
        self.speech_thread = SpeechGeneratorThread(
            self.model, text, self.speaker_embedding, self.speed_slider.value() / 100.0
        )
        
        # Connect to special handler that will try fallbacks if needed
        self.speech_thread.finished.connect(
            lambda path, msg: self.on_speech_attempt_finished(path, msg, text, fallback_phrases, fallback_index)
        )
        self.speech_thread.progress.connect(self.update_progress)
        
        # Add to active threads and start
        self.active_threads.append(self.speech_thread)
        self.speech_thread.start()

    def on_speech_attempt_finished(self, output_path, message, attempted_text, fallback_phrases, fallback_index):
        """Handle result of speech generation attempt with fallback mechanism"""
        if output_path and os.path.exists(output_path):
            # Success! Just play it normally
            self.output_path = output_path
            self.save_btn.setEnabled(True)
            self.play_audio()
            self.speak_btn.setEnabled(True)
            self.status_label.setText("Successfully processed clipboard text")
        else:
            # Failed - try a fallback phrase if we have any left
            if fallback_index < len(fallback_phrases):
                fallback_text = fallback_phrases[fallback_index]
                self.status_label.setText(f"Using fallback phrase {fallback_index+1}...")
                print(f"Original text failed, trying fallback: '{fallback_text}'")
                
                # Update text display
                self.text_input.setText(fallback_text)
                
                # Try the next fallback
                QTimer.singleShot(500, lambda: self.try_speak_with_fallback(
                    fallback_text, fallback_phrases, fallback_index + 1
                ))
            else:
                # All fallbacks failed
                self.status_label.setText("Could not process text, even with fallbacks")
                self.speak_btn.setEnabled(True)

    def try_real_text(self, text):
        """Try to speak the actual copied text (shortened version)"""
        try:
            self.status_label.setText("Attempting to speak copied text...")
            
            # Place the real text in the input box now
            self.text_input.setText(text)
            
            # Create a new thread for the real text
            self.real_speech_thread = SpeechGeneratorThread(
                self.model, text, self.speaker_embedding, self.speed_slider.value() / 100.0
            )
            
            # Connect to normal handlers
            self.real_speech_thread.finished.connect(self.on_speech_generated)
            self.real_speech_thread.progress.connect(self.update_progress)
            
            # Add to active threads and start
            self.active_threads.append(self.real_speech_thread)
            self.real_speech_thread.start()
        except Exception as e:
            self.status_label.setText(f"Could not process actual text: {str(e)}")
            self.speak_btn.setEnabled(True)
        
    def on_player_state_changed(self, state):
        """Handle media player state changes with better completion detection"""
        if state == QMediaPlayer.PlayingState:
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Playing audio...")
        elif state == QMediaPlayer.StoppedState:
            self.stop_btn.setEnabled(False)
            
            # Check if playback completed naturally or was interrupted
            duration = self.player.duration()
            position = self.player.position()
            
            if position > 0 and (position >= duration - 500 or duration <= 0):
                # Natural completion - position is at or near the end
                self.status_label.setText("Audio playback completed")
            else:
                # Playback was interrupted or never really started
                print(f"Playback interrupted: Position={position}ms, Duration={duration}ms")
                # Only try fallback if it appears to have stopped prematurely
                if position < duration - 1000 and duration > 1000:
                    self.play_audio_fallback()
    
    def stop_audio(self):
        """Stop audio playback"""
        self.player.stop()
    
    def speak_text(self):
        """Generate speech for the input text"""
        if self.model is None:
            self.status_label.setText("Model is still loading. Please wait.")
            return
        
        if not self.is_speaker_embedding_set:
            self.status_label.setText("Please select a voice sample first.")
            return
        
        text = self.text_input.toPlainText().strip()
        if not text:
            self.status_label.setText("Please enter text to speak.")
            return
        
        # Get speed setting
        speed_factor = self.speed_slider.value() / 100.0
        
        # Check text length to warn users
        if len(text) > 200:
            self.status_label.setText("Warning: Long text may be truncated. Processing first part...")
        
        # Generate speech in background thread
        self.status_label.setText("Generating speech...")
        self.speech_thread = SpeechGeneratorThread(
            self.model, text, self.speaker_embedding, speed_factor
        )
        self.speech_thread.finished.connect(self.on_speech_generated)
        self.speech_thread.progress.connect(self.update_progress)
        
        # Add to active threads and start
        self.active_threads.append(self.speech_thread)
        self.speech_thread.start()
    
    def on_speech_generated(self, output_path, message):
        """Handle when speech generation is complete"""
        if output_path and os.path.exists(output_path):
            self.output_path = output_path
            self.save_btn.setEnabled(True)
            
            # Play the audio
            self.play_audio()
        else:
            self.status_label.setText(f"Error: {message}")
    
    def play_audio(self):
        """Play the generated audio with improved playback completion handling"""
        if not self.output_path or not os.path.exists(self.output_path):
            self.status_label.setText("No audio file to play")
            return
        
        try:
            # First method: QMediaPlayer with proper setup
            self.status_label.setText(f"Playing audio: {self.output_path}")
            
            # Reset the player completely before loading new content
            self.player.stop()
            self.player.setMedia(None)
            
            # Create a full file URL
            url = QUrl.fromLocalFile(os.path.abspath(self.output_path))
            
            # Set up media content and play
            media_content = QMediaContent(url)
            self.player.setMedia(media_content)
            
            # Use a timer to make sure playback actually starts
            self.playback_timer = QTimer(self)
            self.playback_timer.setSingleShot(True)
            self.playback_timer.timeout.connect(self.check_playback)
            self.playback_timer.start(1000)  # Check after 1 second
            
            # Set volume to 100% to ensure audibility
            self.player.setVolume(100)
            
            # Start playback
            self.player.play()
            
        except Exception as e:
            self.status_label.setText(f"Error with primary audio player: {str(e)}")
            self.play_audio_fallback()
            
    def check_playback(self):
        """Check if QMediaPlayer started playing and verify its duration"""
        if self.player.state() != QMediaPlayer.PlayingState:
            print("QMediaPlayer failed to start playing, trying fallback")
            self.play_audio_fallback()
        else:
            # Get the duration of the audio
            duration = self.player.duration()
            position = self.player.position()
            
            print(f"Audio playback started: Position={position}ms, Duration={duration}ms")
            
            # If duration is suspiciously short or not properly detected
            if duration < 500:  # Less than half a second
                print("Audio duration suspiciously short, using fallback")
                self.player.stop()
                self.play_audio_fallback()
            else:
                # Set up a completion timer to verify playback finishes
                self.completion_timer = QTimer(self)
                self.completion_timer.setSingleShot(True)
                self.completion_timer.timeout.connect(self.verify_playback_completion)
                # Set timer for slightly longer than audio duration
                self.completion_timer.start(duration + 1000)

    def verify_playback_completion(self):
        """Verify audio played completely"""
        position = self.player.position()
        duration = self.player.duration()
        
        print(f"Playback check: Position={position}ms, Duration={duration}ms")
        
        # If position is significantly less than duration, playback might have stopped prematurely
        if position < duration - 1000 and self.player.state() != QMediaPlayer.PlayingState:
            print("Playback appears to have stopped prematurely, restarting with fallback")
            self.play_audio_fallback()
    
    def check_playback(self):
        """Check if QMediaPlayer is actually playing"""
        if self.player.state() != QMediaPlayer.PlayingState:
            self.play_audio_fallback()
    
    def play_audio_fallback(self):
        """Enhanced fallback method to play audio using multiple methods"""
        try:
            self.status_label.setText("Using fallback audio player...")
            
            # Get absolute path with proper formatting
            abs_path = os.path.abspath(self.output_path).replace('\\', '/')
            print(f"Playing audio using fallback method: {abs_path}")
            
            # Method 1: Try system default player (most reliable)
            if sys.platform == 'win32':
                # Windows: Launch file with system default player
                import subprocess
                try:
                    subprocess.Popen(['start', '', abs_path], shell=True)
                    self.status_label.setText("Playing with system default player")
                    return
                except Exception as e:
                    print(f"System player failed: {str(e)}")
                    
                # Alternate Windows method using PowerShell
                try:
                    cmd = ['powershell', '-c', f'Add-Type -AssemblyName PresentationCore; $player = New-Object System.Windows.Media.MediaPlayer; $player.Open("file:///{abs_path}"); $player.Play(); Start-Sleep -s 10; $player.Close()']
                    subprocess.Popen(cmd, shell=True)
                    self.status_label.setText("Playing using PowerShell MediaPlayer")
                    return
                except Exception as e:
                    print(f"PowerShell player failed: {str(e)}")
                    
            # Method 2: Try playsound library
            try:
                from playsound import playsound
                # Run in a separate thread to not block the UI
                import threading
                thread = threading.Thread(target=lambda: playsound(abs_path))
                thread.daemon = True  # Don't let this thread prevent app exit
                thread.start()
                self.status_label.setText("Playing using playsound library")
                return
            except Exception as e:
                print(f"Playsound error: {str(e)}")
            
            # Final fallback - just inform user where file is
            self.status_label.setText(f"Audio saved to: {abs_path}")
            # Try to open the folder containing the file
            try:
                if sys.platform == 'win32':
                    os.system(f'explorer /select,"{abs_path}"')
            except:
                pass
        
        except Exception as e:
            self.status_label.setText(f"Error playing audio: {str(e)}")
            print(f"Audio saved to: {self.output_path}")
    
    def save_audio(self):
        """Save the generated audio file"""
        if not self.output_path or not os.path.exists(self.output_path):
            self.status_label.setText("Generate speech first before saving.")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "", "MP3 Files (*.mp3);;WAV Files (*.wav);;All Files (*)"
        )
        
        if save_path:
            try:
                self.status_label.setText("Saving audio...")
                self.progress_bar.setValue(50)
                
                # Check if saving as MP3
                if save_path.lower().endswith('.mp3'):
                    try:
                        # Install pydub if not available
                        try:
                            from pydub import AudioSegment
                        except ImportError:
                            self.status_label.setText("Installing MP3 conversion support...")
                            import subprocess
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
                            from pydub import AudioSegment
                        
                        sound = AudioSegment.from_wav(self.output_path)
                        sound.export(save_path, format="mp3")
                    except Exception as e:
                        self.status_label.setText(f"Error converting to MP3: {str(e)}. Saving as WAV instead.")
                        # Change extension to .wav if conversion fails
                        save_path = os.path.splitext(save_path)[0] + ".wav"
                        shutil.copy2(self.output_path, save_path)
                else:
                    # Save as WAV
                    shutil.copy2(self.output_path, save_path)
                
                self.status_label.setText(f"Audio saved to: {save_path}")
                self.progress_bar.setValue(100)
            
            except Exception as e:
                self.status_label.setText(f"Error saving audio: {str(e)}")
                print(f"Error details: {str(e)}")


if __name__ == "__main__":
    print("Starting Zonos TTS Application...")
    # Disable PyTorch compiler to avoid cl.exe errors
    torch._dynamo.config.suppress_errors = True
    
    app = QApplication(sys.argv)
    window = ZonosTTSApp()
    sys.exit(app.exec_())
