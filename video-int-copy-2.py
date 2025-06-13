import os
import cv2
import numpy as np
import pandas as pd
import librosa
import moviepy.editor as mp
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from google.cloud import videointelligence_v1 as vi
from google.cloud import vision
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import pysrt
import subprocess
import soundfile as sf
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
from jinja2 import Environment, FileSystemLoader, Template
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import re
from datetime import timedelta
from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import streamlit as st
warnings.filterwarnings('ignore')

# === Setup ===
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vdc200015-ai-media-1-np-8589288c7aa3 1.json'
vision_client = vision.ImageAnnotatorClient()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
use_model = SentenceTransformer('all-mpnet-base-v2')

class InsightGenerator:
    def __init__(self, combined_file_path, model_name="llama3.2:3b"):
        self.combined_file_path = combined_file_path
        self.model = ChatOllama(model=model_name)
        self.df = pd.read_excel(self.combined_file_path)
    
    def build_data_summary(self):
        summary = f"\n=== FILE: combined_analysis ===\n"
        summary += f"Rows: {len(self.df)}, Columns: {len(self.df.columns)}\n"
        summary += f"Column names: {', '.join(self.df.columns)}\n"
        summary += "Sample data:\n"
        for i, row in self.df.head(20).iterrows():
            row_dict = {k: v for k, v in row.items()}
            summary += f"Row {i+1}: {row_dict}\n"
        return summary
    
    def generate_insight(self):
        data_summary = self.build_data_summary()

        system_prompt = """You are analyzing video emotion data. The user has provided Excel files with emotion analysis results.

Generate a professional report with these sections:
1. Executive Summary
2. Emotional Arc Analysis 
3. Dominant Emotions
4. Cross-Modal Analysis
5. Key Insights
6. Recommendations

Focus on the actual data provided below and give specific insights."""

        human_prompt = f"""Here is video emotion analysis data to analyze:

{data_summary}

Please analyze this data and create a concise professional report following the 6-section structure. Be specific about the emotions, patterns, and insights you find in this actual data."""

        response = self.model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

        return response.content
    
class YouTubeDownloader:
    def __init__(self, youtube_url, download_path='./downloads', ffmpeg_path=None):
        self.youtube_url = youtube_url
        self.download_path = download_path
        self.ffmpeg_path = ffmpeg_path or 'ffmpeg'  # default ffmpeg from PATH
        self.video_id = None
        self.title = None
        self.available_langs = []

        os.makedirs(self.download_path, exist_ok=True)

    @staticmethod
    def sanitize_filename(name):
        return re.sub(r'[\\/*?:"<>|]', "", name)

    @staticmethod
    def format_timestamp(seconds):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        millis = int((td.total_seconds() - total_seconds) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"

    def download_video_and_subtitles(self):
        # Probe available subtitles
        probe_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'listsubtitles': True,
        }

        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(self.youtube_url, download=False)
            subs = info.get('subtitles', {})
            auto_subs = info.get('automatic_captions', {})
            self.available_langs = list(subs.keys()) + list(auto_subs.keys())
            print(f"Available subtitle languages: {self.available_langs}")

        if not self.available_langs:
            print("⚠ No subtitles found (manual or auto)")
            return None

        selected_lang = 'en' if 'en' in self.available_langs else self.available_langs[0]

        ydl_opts = {
            'outtmpl': os.path.join(self.download_path, '%(title)s.%(ext)s'),
            'format': 'bestvideo+bestaudio/best',
            'merge_output_format': 'mp4',
            'ffmpeg_location': self.ffmpeg_path,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': [selected_lang],
            'postprocessors': [{'key': 'FFmpegEmbedSubtitle'}],
            'ignoreerrors': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.youtube_url, download=True)
            self.title = info.get("title", "unknown_title")
            self.video_id = info.get("id", None)

        print(f"\n✅ Video downloaded: {self.title}")
        return self.get_video_path()

    def fallback_download_transcript(self):
        try:
            transcript = None
            try_langs = ['en'] + self.available_langs
            for lang in try_langs:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(self.video_id, languages=[lang])
                    print(f"✅ Transcript found using language: {lang}")
                    break
                except Exception:
                    continue

            if not transcript:
                print("❌ No transcript found via YouTubeTranscriptAPI")
                return

            srt_path = os.path.join(self.download_path, f"{self.sanitize_filename(self.title)}.srt")
            with open(srt_path, "w", encoding='utf-8') as f:
                for i, entry in enumerate(transcript, 1):
                    start = entry['start']
                    dur = entry['duration']
                    end = start + dur
                    f.write(f"{i}\n")
                    f.write(f"{self.format_timestamp(start)} --> {self.format_timestamp(end)}\n")
                    f.write(f"{entry['text']}\n\n")
            print(f"✅ Transcript saved: {srt_path}")
        except Exception as e:
            print(f"❌ Failed extracting transcript: {e}")

    def get_video_path(self):
        mp4_files = [f for f in os.listdir(self.download_path) if f.endswith('.mp4')]
        if mp4_files:
            return os.path.join(self.download_path, mp4_files[0])
        return None

    def get_srt_path(self):
        srt_files = [f for f in os.listdir(self.download_path) if f.endswith('.srt')]
        if srt_files:
            return os.path.join(self.download_path, srt_files[0])
        return None
    
class EnhancedVideoSceneAnalyzer:
    def __init__(self, video_path, output_dir="enhanced_video_analysis_Criminal Justice"):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.shots_dir = os.path.join(self.output_dir, "shots")
        self.graphs_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.shots_dir, exist_ok=True)
        os.makedirs(self.graphs_dir, exist_ok=True)
        
        # Data storage
        self.shots = []
        self.shot_audio_emotions = []
        self.shot_visual_emotions = []
        self.shot_person_data = []
        self.overall_analysis = {}
        
        # Get video properties
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

    def detect_shots(self):
        """Detect shots using Google Video Intelligence API"""
        print("🎬 Detecting shots using Google Video Intelligence API...")
        client = vi.VideoIntelligenceServiceClient()
        
        with open(self.video_path, "rb") as f:
            input_content = f.read()
        
        features = [vi.Feature.SHOT_CHANGE_DETECTION]
        operation = client.annotate_video(
            request={"features": features, "input_content": input_content}
        )
        result = operation.result(timeout=300)
        shots = result.annotation_results[0].shot_annotations
        
        for i, shot in enumerate(shots):
            start = shot.start_time_offset.seconds + shot.start_time_offset.microseconds / 1e6
            end = shot.end_time_offset.seconds + shot.end_time_offset.microseconds / 1e6
            duration = end - start
            
            self.shots.append({
                'shot_id': i,
                'start_time': start,
                'end_time': end,
                'duration': duration,
                'start_frame': int(start * self.fps),
                'end_frame': int(end * self.fps)
            })
        
        print(f"✅ Detected {len(self.shots)} shots")
        return self.shots

    def extract_shot_keyframes(self):
        """Extract multiple keyframes per shot for better analysis"""
        print("🖼️ Extracting keyframes for each shot...")
        
        for shot in self.shots:
            shot_id = shot['shot_id']
            start_frame = shot['start_frame']
            end_frame = shot['end_frame']
            
            # Extract 3 keyframes per shot: beginning, middle, end
            keyframe_positions = [
                start_frame,
                (start_frame + end_frame) // 2,
                max(start_frame + 1, end_frame - 1)
            ]
            
            keyframes = []
            for pos_idx, frame_pos in enumerate(keyframe_positions):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = self.cap.read()
                if ret:
                    keyframe_path = os.path.join(
                        self.shots_dir, 
                        f"shot_{shot_id:03d}_keyframe_{pos_idx}.jpg"
                    )
                    cv2.imwrite(keyframe_path, frame)
                    keyframes.append(keyframe_path)
            
            shot['keyframes'] = keyframes
        
        print("✅ Keyframes extracted")

    def analyze_shot_audio_emotions(self):
        """Analyze audio emotions for each shot"""
        print("🎵 Analyzing audio emotions for each shot...")
        
        # Extract full audio
        audio_path = os.path.join(self.output_dir, "full_audio.wav")
        video = mp.VideoFileClip(self.video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
        
        # Load audio
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        
        for shot in self.shots:
            start_sample = int(shot['start_time'] * sample_rate)
            end_sample = int(shot['end_time'] * sample_rate)
            shot_audio = audio_data[start_sample:end_sample]
            
            # Extract features
            features = self._extract_emotion_features(shot_audio, sample_rate)
            emotion = self._classify_emotion(features)
            
            shot_emotion_data = {
                'shot_id': shot['shot_id'],
                'start_time': shot['start_time'],
                'end_time': shot['end_time'],
                'duration': shot['duration'],
                'audio_emotion': emotion,
                'audio_intensity': self._emotion_to_intensity(emotion),
                **features
            }
            
            self.shot_audio_emotions.append(shot_emotion_data)
        
        print("✅ Audio emotion analysis complete")

    def analyze_shot_visual_emotions(self):
        """Analyze visual emotions and output scores in a format matching the emotions_result.csv template"""
        print("👁️ Analyzing visual emotions and formatting results as per template...")

        emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self.shot_visual_emotions = []

        for shot in self.shots:
            shot_id = shot['shot_id']
            start_time = shot['start_time']
            end_time = shot['end_time']
            shot_emotions_all = []
            total_persons = 0

            for keyframe_path in shot.get('keyframes', []):
                if not os.path.exists(keyframe_path):
                    continue

                frame = cv2.imread(keyframe_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for i, (x, y, w, h) in enumerate(faces):
                    total_persons += 1

                    # Generate mock emotion distribution (or plug in real model here)
                    np.random.seed(shot_id * 100 + i)
                    emotion_scores = np.random.dirichlet(np.ones(len(emotion_labels)) * 0.8)
                    emotion_dict = {emotion: float(score) for emotion, score in zip(emotion_labels, emotion_scores)}
                    shot_emotions_all.append(emotion_dict)

            # Aggregate emotions
            if total_persons > 0 and shot_emotions_all:
                df = pd.DataFrame(shot_emotions_all)
                emotion_means = df.mean().to_dict()
                dominant_emotion = max(emotion_means, key=emotion_means.get)
            else:
                emotion_means = {e: 0.0 for e in emotion_labels}
                dominant_emotion = None  # or you can use 'no_face_detected'

            # Convert to output structure
            shot_data = {
                "shot_id": shot_id,
                "start_time": start_time,
                "end_time": end_time,
                "total_persons_detected": total_persons,
                "dominant_visual_emotion": dominant_emotion,
                **emotion_means
            }

            self.shot_visual_emotions.append(shot_data)

        print(f"✅ Generated visual emotion results for {len(self.shot_visual_emotions)} shots in template format")

    # def analyze_shot_visual_emotions(self):
    #     """Analyze visual emotions and output scores in a format matching the emotions_result.csv template"""
    #     print("👁️ Analyzing visual emotions and formatting results as per template...")

    #     emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    #     self.shot_visual_emotions = []

    #     for shot in self.shots:
    #         shot_id = shot['shot_id']
    #         start_time = shot['start_time']
    #         end_time = shot['end_time']
    #         shot_emotions_all = []
    #         total_persons = 0

    #         for keyframe_path in shot.get('keyframes', []):
    #             if not os.path.exists(keyframe_path):
    #                 continue

    #             frame = cv2.imread(keyframe_path)
    #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #             face_cascade = cv2.CascadeClassifier(
    #                 cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    #             )
    #             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    #             for i, (x, y, w, h) in enumerate(faces):
    #                 total_persons += 1

    #                 # Generate mock emotion distribution (or plug in real model here)
    #                 np.random.seed(shot_id * 100 + i)
    #                 emotion_scores = np.random.dirichlet(np.ones(len(emotion_labels)) * 0.8)
    #                 emotion_dict = {emotion: float(score) for emotion, score in zip(emotion_labels, emotion_scores)}
    #                 shot_emotions_all.append(emotion_dict)

    #         # Aggregate emotions
    #         if shot_emotions_all:
    #             df = pd.DataFrame(shot_emotions_all)
    #             emotion_means = df.mean().to_dict()
    #             dominant_emotion = max(emotion_means, key=emotion_means.get)
    #         else:
    #             emotion_means = {e: 0.0 for e in emotion_labels}
    #             dominant_emotion = "neutral"

    #         # Convert to output structure
    #         shot_data = {
    #             "shot_id": shot_id,
    #             "start_time": start_time,
    #             "end_time": end_time,
    #             "total_persons_detected": total_persons,
    #             "dominant_visual_emotion": dominant_emotion,
    #             **emotion_means
    #         }

    #         self.shot_visual_emotions.append(shot_data)

    #     print(f"✅ Generated visual emotion results for {len(self.shot_visual_emotions)} shots in template format")

    def run_clip_analysis(self):
        """Run CLIP analysis on shots with comprehensive prompts"""
        print("🔍 Running CLIP analysis on shots...")
        
        text_prompts = [
            "happy family moment", "sad emotional scene", "action packed sequence",
            "romantic couple", "horror scary scene", "peaceful calm moment",
            "angry confrontation", "suspenseful tense moment", "comedy funny scene",
            "dramatic intense scene", "children playing", "celebration party",
            "conflict argument", "surprise shocking moment"
        ]
        
        clip_results = []
        
        for shot in self.shots:
            shot_id = shot['shot_id']
            
            # Use middle keyframe for CLIP analysis
            if len(shot.get('keyframes', [])) > 1:
                keyframe_path = shot['keyframes'][1]  # middle keyframe
            elif shot.get('keyframes'):
                keyframe_path = shot['keyframes'][0]
            else:
                continue
            
            if not os.path.exists(keyframe_path):
                continue
            
            # Load and process image
            image = cv2.imread(keyframe_path)[:, :, ::-1]  # BGR to RGB
            inputs = clip_processor(
                text=text_prompts, 
                images=image, 
                return_tensors="pt", 
                padding=True
            )
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image.detach().numpy().flatten()
            
            # Get top matches
            prompt_scores = list(zip(text_prompts, logits_per_image))
            prompt_scores.sort(key=lambda x: x[1], reverse=True)
            
            result = {
                'shot_id': shot_id,
                'start_time': shot['start_time'],
                'end_time': shot['end_time'],
                'top_scene_type': prompt_scores[0][0],
                'top_score': float(prompt_scores[0][1]),
                'scene_scores': {prompt: float(score) for prompt, score in prompt_scores}
            }
            clip_results.append(result)
        
        # Save CLIP results
        df = pd.DataFrame(clip_results)
        df.to_excel(os.path.join(self.output_dir, "clip_scene_analysis.xlsx"), index=False)
        print("✅ CLIP analysis complete")
        
        return clip_results

    def _extract_emotion_features(self, y, sr):
        """Extract comprehensive audio features for emotion analysis"""
        if len(y) == 0:
            return self._get_default_features()
        
        try:
            # Tempo and rhythm
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean()
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
            
            # Energy features
            rms = librosa.feature.rms(y=y)[0].mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0].mean()
            
            # Harmonic analysis
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.mean(y_harmonic**2) / (np.mean(y_percussive**2) + 1e-6)
            
            # Chroma and tonality
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_std = np.std(chroma)
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # Roughness estimation
            spec = np.abs(librosa.stft(y))
            spec_flux = np.mean(np.diff(spec, axis=1)**2) if spec.shape[1] > 1 else 0
            
            return {
                'tempo': tempo,
                'spectral_centroid': spectral_centroid,
                'spectral_bandwidth': spectral_bandwidth,
                'spectral_rolloff': spectral_rolloff,
                'spectral_contrast': spectral_contrast,
                'rms': rms,
                'zero_crossing_rate': zero_crossing_rate,
                'harmonic_ratio': harmonic_ratio,
                'chroma_std': chroma_std,
                'spec_flux': spec_flux,
                'mfcc_1': mfcc_mean[0] if len(mfcc_mean) > 0 else 0,
                'mfcc_2': mfcc_mean[1] if len(mfcc_mean) > 1 else 0,
                'mfcc_3': mfcc_mean[2] if len(mfcc_mean) > 2 else 0
            }
        except Exception as e:
            print(f"Warning: Feature extraction failed for audio segment: {e}")
            return self._get_default_features()
    
    def _get_default_features(self):
        """Return default features when extraction fails"""
        return {
            'tempo': 120.0, 'spectral_centroid': 1000.0, 'spectral_bandwidth': 1000.0,
            'spectral_rolloff': 2000.0, 'spectral_contrast': 0.5, 'rms': 0.1,
            'zero_crossing_rate': 0.1, 'harmonic_ratio': 1.0, 'chroma_std': 0.5,
            'spec_flux': 0.1, 'mfcc_1': 0.0, 'mfcc_2': 0.0, 'mfcc_3': 0.0
        }

    def _classify_emotion(self, features):
        """Enhanced emotion classification based on audio features"""
        tempo = features['tempo']
        rms = features['rms']
        spectral_centroid = features['spectral_centroid']
        spectral_contrast = features['spectral_contrast']
        harmonic_ratio = features['harmonic_ratio']
        zero_crossing_rate = features['zero_crossing_rate']
        
        # Multi-criteria emotion classification
        if tempo > 140 and rms > 0.15:
            if spectral_contrast > 6 and harmonic_ratio < 0.3:
                return "horror"
            elif spectral_centroid > 3000:
                return "exciting"
            else:
                return "energetic"
        elif tempo > 120 and rms > 0.1:
            if zero_crossing_rate > 0.15:
                return "happy"
            else:
                return "upbeat"
        elif tempo < 80:
            if rms < 0.05:
                return "melancholic"
            elif spectral_contrast < 4:
                return "peaceful"
            else:
                return "dramatic"
        elif rms > 0.2:
            return "intense"
        else:
            return "neutral"

    def _emotion_to_intensity(self, emotion):
        """Map emotions to intensity scores"""
        intensity_map = {
            'peaceful': 1, 'melancholic': 2, 'neutral': 3, 'happy': 4, 
            'upbeat': 5, 'dramatic': 6, 'intense': 7, 'exciting': 8, 
            'energetic': 8, 'horror': 9
        }
        return intensity_map.get(emotion, 3)

    def generate_comprehensive_analysis(self):
        """Generate overall video emotion analysis"""
        print("📊 Generating comprehensive analysis...")
        
        # Audio emotion distribution
        audio_emotions = [shot['audio_emotion'] for shot in self.shot_audio_emotions]
        audio_emotion_counts = Counter(audio_emotions)
        dominant_audio_emotion = max(audio_emotion_counts, key=audio_emotion_counts.get)
        
        # Visual emotion distribution
        # Visual emotion distribution (filter out None values)
        visual_emotions = [shot['dominant_visual_emotion'] for shot in self.shot_visual_emotions if shot['dominant_visual_emotion'] is not None]
        visual_emotion_counts = Counter(visual_emotions)

        if visual_emotion_counts:
            dominant_visual_emotion = max(visual_emotion_counts, key=visual_emotion_counts.get)
        else:
            dominant_visual_emotion = "no_visual_dominant"

        
        # Overall intensity analysis
        intensities = [shot['audio_intensity'] for shot in self.shot_audio_emotions]
        avg_intensity = np.mean(intensities)
        
        # Person analysis
        total_persons = sum(shot['total_persons_detected'] for shot in self.shot_visual_emotions)
        shots_with_persons = sum(1 for shot in self.shot_visual_emotions if shot['total_persons_detected'] > 0)
        person_density = shots_with_persons / len(self.shots) if self.shots else 0
        
        # Generate overall emotion statement
        emotion_statement = self._generate_emotion_statement(
            dominant_audio_emotion, dominant_visual_emotion, avg_intensity, person_density
        )
        
        self.overall_analysis = {
            'total_shots': len(self.shots),
            'total_duration': self.duration,
            'dominant_audio_emotion': dominant_audio_emotion,
            'dominant_visual_emotion': dominant_visual_emotion,
            'average_intensity': avg_intensity,
            'total_persons_detected': total_persons,
            'person_density': person_density,
            'audio_emotion_distribution': dict(audio_emotion_counts),
            'visual_emotion_distribution': dict(visual_emotion_counts),
            'emotion_statement': emotion_statement
        }
        
        # Save comprehensive analysis
        analysis_df = pd.DataFrame([self.overall_analysis])
        analysis_df.to_excel(os.path.join(self.output_dir, "overall_analysis.xlsx"), index=False)
        
        print("✅ Comprehensive analysis complete")

    def _generate_emotion_statement(self, audio_emotion, visual_emotion, intensity, person_density):
        """Generate a descriptive emotion statement for the video"""
        intensity_desc = "low" if intensity < 4 else ("moderate" if intensity < 7 else "high")
        person_desc = "intimate" if person_density < 0.3 else ("social" if person_density < 0.7 else "crowded")
        
        statement = f"This video presents a {intensity_desc}-intensity emotional experience, "
        statement += f"characterized primarily by {audio_emotion} audio tones and {visual_emotion} visual expressions. "
        statement += f"The content appears to be {person_desc} in nature, "
        
        if audio_emotion == visual_emotion:
            statement += f"with consistent {audio_emotion} emotional messaging across both audio and visual elements."
        else:
            statement += f"creating an interesting contrast between {audio_emotion} audio and {visual_emotion} visual emotions."
        
        return statement

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("📈 Creating detailed visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Shot-wise emotion timeline
        self._create_emotion_timeline()
        
        # 2. Emotion distribution charts
        self._create_emotion_distributions()
        
        # 3. Person detection analysis
        self._create_person_analysis()
        
        # 4. Audio intensity heatmap
        self._create_intensity_heatmap()
        
        # 5. Comprehensive dashboard
        self._create_dashboard()
        
        print("✅ All visualizations created")

    def _create_emotion_timeline(self):
        """Create shot-wise emotion timeline"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Audio emotions timeline
        shot_times = [shot['start_time'] for shot in self.shot_audio_emotions]
        audio_emotions = [shot['audio_emotion'] for shot in self.shot_audio_emotions]
        intensities = [shot['audio_intensity'] for shot in self.shot_audio_emotions]
        
        emotion_colors = {
            'peaceful': '#90EE90', 'melancholic': '#4682B4', 'neutral': '#D3D3D3',
            'happy': '#FFD700', 'upbeat': '#FFA500', 'dramatic': '#800080',
            'intense': '#DC143C', 'exciting': '#FF4500', 'energetic': '#FF6347',
            'horror': '#000000'
        }
        
        colors = [emotion_colors.get(emotion, '#808080') for emotion in audio_emotions]
        ax1.scatter(shot_times, intensities, c=colors, s=100, alpha=0.7)
        ax1.plot(shot_times, intensities, 'k-', alpha=0.3)
        ax1.set_ylabel('Audio Intensity')
        ax1.set_title('Shot-wise Audio Emotion Timeline')
        ax1.grid(True, alpha=0.3)
        
        # Visual emotions timeline
        visual_shot_times = [shot['start_time'] for shot in self.shot_visual_emotions]
        visual_emotions = [shot['dominant_visual_emotion'] for shot in self.shot_visual_emotions]
        visual_colors = [emotion_colors.get(emotion, '#808080') for emotion in visual_emotions]
        
        ax2.scatter(visual_shot_times, [1]*len(visual_shot_times), c=visual_colors, s=150, alpha=0.7)
        ax2.set_ylabel('Visual Emotions')
        ax2.set_ylim(0.5, 1.5)
        ax2.set_yticks([1])
        ax2.set_yticklabels(['Visual'])
        ax2.grid(True, alpha=0.3)
        
        # Person count timeline
        person_counts = [shot['total_persons_detected'] for shot in self.shot_visual_emotions]
        ax3.bar(visual_shot_times, person_counts, width=2, alpha=0.6, color='skyblue')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Person Count')
        ax3.set_title('Person Detection Timeline')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'emotion_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_emotion_distributions(self):
        """Create emotion distribution charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Audio emotion pie chart
        audio_emotions = [shot['audio_emotion'] for shot in self.shot_audio_emotions]
        audio_counts = Counter(audio_emotions)
        ax1.pie(audio_counts.values(), labels=audio_counts.keys(), autopct='%1.1f%%', startangle=90)
        ax1.set_title('Audio Emotion Distribution')
        
        # Visual emotion pie chart
        visual_emotions = [shot['dominant_visual_emotion'] for shot in self.shot_visual_emotions]
        visual_counts = Counter(visual_emotions)
        ax2.pie(visual_counts.values(), labels=visual_counts.keys(), autopct='%1.1f%%', startangle=90)
        ax2.set_title('Visual Emotion Distribution')
        
        # Combined emotion bar chart
        all_emotions = set(audio_emotions + visual_emotions)
        audio_vals = [audio_counts.get(emotion, 0) for emotion in all_emotions]
        visual_vals = [visual_counts.get(emotion, 0) for emotion in all_emotions]
        
        x = np.arange(len(all_emotions))
        width = 0.35
        ax3.bar(x - width/2, audio_vals, width, label='Audio', alpha=0.8)
        ax3.bar(x + width/2, visual_vals, width, label='Visual', alpha=0.8)
        ax3.set_xlabel('Emotions')
        ax3.set_ylabel('Count')
        ax3.set_title('Audio vs Visual Emotion Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(all_emotions, rotation=45)
        ax3.legend()
        
        # Intensity distribution
        intensities = [shot['audio_intensity'] for shot in self.shot_audio_emotions]
        ax4.hist(intensities, bins=10, alpha=0.7, color='coral', edgecolor='black')
        ax4.set_xlabel('Intensity Level')
        ax4.set_ylabel('Number of Shots')
        ax4.set_title('Audio Intensity Distribution')
        ax4.axvline(np.mean(intensities), color='red', linestyle='--', label=f'Mean: {np.mean(intensities):.1f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'emotion_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_person_analysis(self):
        """Create person detection analysis charts"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Person count distribution
        person_counts = [shot['total_persons_detected'] for shot in self.shot_visual_emotions]
        count_dist = Counter(person_counts)
        ax1.bar(count_dist.keys(), count_dist.values(), alpha=0.7, color='lightblue', edgecolor='navy')
        ax1.set_xlabel('Number of Persons Detected')
        ax1.set_ylabel('Number of Shots')
        ax1.set_title('Person Count Distribution Across Shots')
        ax1.grid(True, alpha=0.3)
        
        # Person presence over time
        shot_times = [shot['start_time'] for shot in self.shot_visual_emotions]
        ax2.plot(shot_times, person_counts, 'o-', color='navy', linewidth=2, markersize=6)
        ax2.fill_between(shot_times, person_counts, alpha=0.3, color='lightblue')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Person Count')
        ax2.set_title('Person Presence Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'person_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_intensity_heatmap(self):
        """Create audio intensity heatmap"""
        # Create intensity matrix for heatmap
        n_shots = len(self.shot_audio_emotions)
        features = ['tempo', 'rms', 'spectral_centroid', 'spectral_contrast', 'harmonic_ratio']
        
        intensity_matrix = np.zeros((len(features), n_shots))
        for i, shot in enumerate(self.shot_audio_emotions):
            for j, feature in enumerate(features):
                intensity_matrix[j, i] = shot.get(feature, 0)
        
        # Normalize features
        scaler = StandardScaler()
        intensity_matrix_norm = scaler.fit_transform(intensity_matrix)
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(intensity_matrix_norm, 
                   xticklabels=[f"Shot {i}" for i in range(n_shots)],
                   yticklabels=features,
                   cmap='RdYlBu_r', center=0, annot=False)
        plt.title('Audio Feature Intensity Heatmap Across Shots')
        plt.xlabel('Shots')
        plt.ylabel('Audio Features')
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'intensity_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _create_dashboard(self):
        """Create comprehensive dashboard"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Video Emotion Analysis Dashboard\n{os.path.basename(self.video_path)}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Audio Emotion Distribution (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        audio_emotions = [shot['audio_emotion'] for shot in self.shot_audio_emotions]
        audio_counts = Counter(audio_emotions)
        wedges, texts, autotexts = ax1.pie(audio_counts.values(), labels=audio_counts.keys(), 
                                          autopct='%1.1f%%', startangle=90)
        ax1.set_title('Audio Emotions', fontweight='bold')
        
        # 2. Visual Emotion Distribution (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        visual_emotions = [shot['dominant_visual_emotion'] for shot in self.shot_visual_emotions]
        visual_counts = Counter(visual_emotions)
        ax2.pie(visual_counts.values(), labels=visual_counts.keys(), 
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Visual Emotions', fontweight='bold')
        
        # 3. Overall Stats (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.axis('off')
        stats_text = f"""
        Total Duration: {self.duration:.1f}s
        Total Shots: {len(self.shots)}
        Avg Shot Length: {self.duration/len(self.shots):.1f}s
        
        Dominant Audio: {self.overall_analysis.get('dominant_audio_emotion', 'N/A')}
        Dominant Visual: {self.overall_analysis.get('dominant_visual_emotion', 'N/A')}
        Avg Intensity: {self.overall_analysis.get('average_intensity', 0):.1f}/10
        
        Total Persons: {self.overall_analysis.get('total_persons_detected', 0)}
        Person Density: {self.overall_analysis.get('person_density', 0):.1%}
        """
        ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax3.set_title('Key Statistics', fontweight='bold')
        
        # 4. Intensity Timeline (second row, spanning 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        shot_times = [shot['start_time'] for shot in self.shot_audio_emotions]
        intensities = [shot['audio_intensity'] for shot in self.shot_audio_emotions]
        ax4.plot(shot_times, intensities, 'o-', linewidth=2, markersize=6, color='crimson')
        ax4.fill_between(shot_times, intensities, alpha=0.3, color='pink')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Intensity Level')
        ax4.set_title('Audio Intensity Over Time', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Person Count Timeline (second row, right)
        ax5 = fig.add_subplot(gs[1, 2])
        visual_times = [shot['start_time'] for shot in self.shot_visual_emotions]
        person_counts = [shot['total_persons_detected'] for shot in self.shot_visual_emotions]
        ax5.bar(visual_times, person_counts, width=2, alpha=0.6, color='skyblue')
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Person Count')
        ax5.set_title('Person Detection', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Feature Correlation (third row, left)
        ax6 = fig.add_subplot(gs[2, 0])
        features = ['tempo', 'rms', 'spectral_centroid', 'spectral_contrast', 'audio_intensity']
        feature_data = []
        for shot in self.shot_audio_emotions:
            row = [shot.get(f, 0) for f in features]
            feature_data.append(row)
        
        if feature_data:
            corr_matrix = np.corrcoef(np.array(feature_data).T)
            im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax6.set_xticks(range(len(features)))
            ax6.set_yticks(range(len(features)))
            ax6.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=8)
            ax6.set_yticklabels([f.replace('_', '\n') for f in features], fontsize=8)
            ax6.set_title('Feature Correlation', fontweight='bold')
            plt.colorbar(im, ax=ax6, shrink=0.8)
        
        # 7. Emotion Comparison (third row, center)
        ax7 = fig.add_subplot(gs[2, 1])
        all_emotions = set(audio_emotions + visual_emotions)
        audio_vals = [audio_counts.get(emotion, 0) for emotion in all_emotions]
        visual_vals = [visual_counts.get(emotion, 0) for emotion in all_emotions]
        
        x = np.arange(len(all_emotions))
        width = 0.35
        ax7.bar(x - width/2, audio_vals, width, label='Audio', alpha=0.8, color='orange')
        ax7.bar(x + width/2, visual_vals, width, label='Visual', alpha=0.8, color='blue')
        ax7.set_xlabel('Emotions')
        ax7.set_ylabel('Count')
        ax7.set_title('Audio vs Visual Comparison', fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(all_emotions, rotation=45, ha='right')
        ax7.legend()
        
        # 8. Intensity Distribution (third row, right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.hist(intensities, bins=8, alpha=0.7, color='coral', edgecolor='black')
        ax8.axvline(np.mean(intensities), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(intensities):.1f}')
        ax8.set_xlabel('Intensity Level')
        ax8.set_ylabel('Number of Shots')
        ax8.set_title('Intensity Distribution', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Emotion Statement (bottom row, spanning all columns)
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')
        emotion_text = self.overall_analysis.get('emotion_statement', 'Analysis in progress...')
        ax9.text(0.5, 0.5, emotion_text, transform=ax9.transAxes, fontsize=14,
                ha='center', va='center', wrap=True,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gold"))
        ax9.set_title('Overall Emotion Analysis', fontweight='bold', pad=20)
        
        plt.savefig(os.path.join(self.graphs_dir, 'comprehensive_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def save_detailed_reports(self):
        """Save detailed Excel reports"""
        print("💾 Saving detailed reports...")
        
        # Shot-level analysis
        shots_df = pd.DataFrame(self.shots)
        shots_df.to_excel(os.path.join(self.output_dir, "shots_analysis.xlsx"), index=False)
        
        # Audio emotion analysis
        audio_df = pd.DataFrame(self.shot_audio_emotions)
        audio_df.to_excel(os.path.join(self.output_dir, "audio_emotions.xlsx"), index=False)
        
        # Visual emotion analysis
        visual_df = pd.DataFrame(self.shot_visual_emotions)
        visual_df.to_excel(os.path.join(self.output_dir, "visual_emotions.xlsx"), index=False)
        
        # Person detection data
        person_df = pd.DataFrame(self.shot_person_data)
        person_df.to_excel(os.path.join(self.output_dir, "person_detection.xlsx"), index=False)
        
        # Combined analysis
        combined_data = []
        for i, shot in enumerate(self.shots):
            audio_data = self.shot_audio_emotions[i] if i < len(self.shot_audio_emotions) else {}
            visual_data = self.shot_visual_emotions[i] if i < len(self.shot_visual_emotions) else {}
            
            combined_row = {
                **shot,
                'audio_emotion': audio_data.get('audio_emotion', 'unknown'),
                'audio_intensity': audio_data.get('audio_intensity', 0),
                'visual_emotion': visual_data.get('dominant_visual_emotion', 'unknown'),
                'persons_detected': visual_data.get('total_persons_detected', 0),
                'tempo': audio_data.get('tempo', 0),
                'rms': audio_data.get('rms', 0),
                'spectral_centroid': audio_data.get('spectral_centroid', 0)
            }
            combined_data.append(combined_row)
        
        combined_df = pd.DataFrame(combined_data)
        combined_df.to_excel(os.path.join(self.output_dir, "combined_analysis.xlsx"), index=False)
        
        print("✅ All reports saved")

    
    def analyze_dialogue_emotions(self):
        """Analyze dialogue emotions using speech-to-text and sentiment analysis, and extract transcript per shot"""
        print("🗣️ Analyzing dialogue emotions and extracting transcripts...")

        # Extract audio for speech analysis
        audio_path = os.path.join(self.output_dir, "dialogue_audio.wav")
        video = mp.VideoFileClip(self.video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)

        # Load audio for analysis
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)

        # Attempt to load corresponding .srt subtitle file
        srt_path = self.video_path.replace(".mp4", ".srt")
        if os.path.exists(srt_path):
            subs = pysrt.open(srt_path)
        else:
            print("⚠️ No subtitles found, continuing without transcript extraction.")
            subs = []

        self.dialogue_emotions = []

        for shot in self.shots:
            start_sample = int(shot['start_time'] * sample_rate)
            end_sample = int(shot['end_time'] * sample_rate)
            shot_audio = audio_data[start_sample:end_sample]

            # === Dialogue Emotion Analysis (simulated) ===
            dialogue_emotion = self._analyze_dialogue_segment(shot_audio, sample_rate)

            # === Transcript Extraction from subtitles ===
            transcript_lines = []
            if subs:
                start = timedelta(seconds=shot['start_time'])
                end = timedelta(seconds=shot['end_time'])
                for sub in subs:
                    sub_start = sub.start.to_time()
                    sub_end = sub.end.to_time()
                    if start <= timedelta(hours=sub_end.hour, minutes=sub_end.minute, seconds=sub_end.second) and \
                    end >= timedelta(hours=sub_start.hour, minutes=sub_start.minute, seconds=sub_start.second):
                        transcript_lines.append(sub.text.strip())
            transcript_text = " ".join(transcript_lines)

            # Store result
            dialogue_data = {
                'shot_id': shot['shot_id'],
                'start_time': shot['start_time'],
                'end_time': shot['end_time'],
                'dialogue_emotion': dialogue_emotion['emotion'],
                'dialogue_intensity': dialogue_emotion['intensity'],
                'confidence': dialogue_emotion['confidence'],
                'dialogue_transcript': transcript_text
            }

            shot['dialogue_transcript'] = transcript_text  # Add to shot as well
            self.dialogue_emotions.append(dialogue_data)

    print("✅ Dialogue emotion + transcript extraction complete.")


    def analyze_music_segments(self):
        """Analyze music segments separately from dialogue"""
        print("🎵 Analyzing music segment emotions...")
        
        # This would typically involve source separation to isolate music
        # For now, we'll simulate music emotion analysis
        self.music_emotions = []
        
        # Extract audio
        audio_path = os.path.join(self.output_dir, "music_audio.wav")
        video = mp.VideoFileClip(self.video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
        
        audio_data, sample_rate = librosa.load(audio_path, sr=None)
        
        # Analyze music in segments (every 2 seconds for detailed analysis)
        segment_length = 2.0  # seconds
        total_segments = int(self.duration / segment_length)
        
        for i in range(total_segments):
            start_time = i * segment_length
            end_time = min((i + 1) * segment_length, self.duration)
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio_data[start_sample:end_sample]
            
            # Extract music-specific features
            music_emotion = self._analyze_music_segment(segment_audio, sample_rate)
            
            music_data = {
                'segment_id': i,
                'start_time': start_time,
                'end_time': end_time,
                'music_emotion': music_emotion['emotion'],
                'music_intensity': music_emotion['intensity'],
                'tempo': music_emotion['tempo'],
                'energy': music_emotion['energy']
            }
            self.music_emotions.append(music_data)
        
        print("✅ Music emotion analysis complete")

    def _analyze_dialogue_segment(self, audio_segment, sample_rate):
        """Analyze dialogue emotion from audio segment"""
        if len(audio_segment) == 0:
            return {'emotion': 'peaceful', 'intensity': 1, 'confidence': 0.5}
        
        # Extract features indicative of speech emotion
        rms = librosa.feature.rms(y=audio_segment)[0].mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)[0].mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)[0].mean()
        
        # Classify dialogue emotion based on speech characteristics
        if rms > 0.2 and spectral_centroid > 3000:
            if zero_crossing_rate > 0.15:
                emotion = 'exciting'
                intensity = 8
            else:
                emotion = 'angry'
                intensity = 6
        elif rms > 0.15:
            emotion = 'dramatic'
            intensity = 7
        elif spectral_centroid > 2000:
            emotion = 'happy'
            intensity = 5
        elif rms < 0.05:
            if spectral_centroid < 1000:
                emotion = 'sad'
                intensity = 3
            else:
                emotion = 'peaceful'
                intensity = 1
        else:
            # Check for horror characteristics
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sample_rate).mean()
            if spectral_contrast > 25 and rms > 0.1:
                emotion = 'horror'
                intensity = 9
            else:
                emotion = 'suspenseful'
                intensity = 4
        
        confidence = min(0.9, max(0.3, rms * 5))  # Confidence based on signal strength
        
        return {'emotion': emotion, 'intensity': intensity, 'confidence': confidence}

    def _analyze_music_segment(self, audio_segment, sample_rate):
        """Analyze music emotion from audio segment"""
        if len(audio_segment) == 0:
            return {'emotion': 'peaceful', 'intensity': 1, 'tempo': 120, 'energy': 0.5}
        
        # Extract music-specific features
        onset_env = librosa.onset.onset_strength(y=audio_segment, sr=sample_rate)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)[0] if len(onset_env) > 0 else 120
        
        # Harmonic and percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(audio_segment)
        harmonic_energy = np.mean(y_harmonic**2)
        percussive_energy = np.mean(y_percussive**2)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sample_rate)[0].mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sample_rate)[0].mean()
        
        # Energy and dynamics
        rms = librosa.feature.rms(y=audio_segment)[0].mean()
        energy = float(rms)
        
        # Classify music emotion
        if tempo > 140 and energy > 0.15:
            emotion = 'exciting'
            intensity = 8
        elif tempo > 120 and harmonic_energy > percussive_energy:
            if spectral_centroid > 2000:
                emotion = 'happy'
                intensity = 7
            else:
                emotion = 'dramatic'
                intensity = 6
        elif tempo < 80:
            if energy < 0.05:
                emotion = 'sad'
                intensity = 3
            else:
                emotion = 'peaceful'
                intensity = 1
        elif spectral_bandwidth > 2000 and energy > 0.2:
            emotion = 'horror'
            intensity = 9
        elif energy > 0.1:
            emotion = 'suspenseful'
            intensity = 5
        else:
            emotion = 'angry'
            intensity = 4
        
        return {
            'emotion': emotion, 
            'intensity': intensity, 
            'tempo': float(tempo), 
            'energy': energy
        }

    def create_dialogue_emotion_timeline(self):
        """Create dialogue emotion intensity timeline"""
        plt.figure(figsize=(16, 6))
        
        # Extract data
        segment_times = [d['start_time'] for d in self.dialogue_emotions]
        emotions = [d['dialogue_emotion'] for d in self.dialogue_emotions]
        intensities = [d['dialogue_intensity'] for d in self.dialogue_emotions]
        
        # Color mapping for emotions
        emotion_colors = {
            'peaceful': '#90EE90', 'sad': '#4682B4', 'happy': '#FFD700',
            'exciting': '#FF4500', 'dramatic': '#800080', 'suspenseful': '#FFA500',
            'angry': '#DC143C', 'horror': '#000000'
        }
        
        colors = [emotion_colors.get(emotion, '#808080') for emotion in emotions]
        
        # Create scatter plot
        plt.scatter(segment_times, intensities, c=colors, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add connecting lines
        plt.plot(segment_times, intensities, 'k-', alpha=0.3, linewidth=1)
        
        # Customize plot
        plt.xlabel('Segment Index (Seconds)', fontsize=12)
        plt.ylabel('Emotion Intensity Score', fontsize=12)
        plt.title('Dialogue Emotion Intensity Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 10)
        
        # Create legend
        unique_emotions = list(set(emotions))
        legend_elements = [plt.scatter([], [], c=emotion_colors.get(emotion, '#808080'), 
                                     s=80, label=emotion) for emotion in unique_emotions]
        plt.legend(handles=legend_elements, title='Emotion Intensity', 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'dialogue_emotion_timeline.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_music_emotion_distribution(self):
        """Create music emotion distribution pie chart"""
        plt.figure(figsize=(10, 8))
        
        # Extract music emotions
        music_emotions = [m['music_emotion'] for m in self.music_emotions]
        emotion_counts = Counter(music_emotions)
        
        # Create pie chart
        colors = ['#FF9500', '#FF4500', '#32CD32', '#DC143C', '#9370DB', 
                 '#4682B4', '#8B4513', '#808080']
        
        wedges, texts, autotexts = plt.pie(emotion_counts.values(), 
                                          labels=emotion_counts.keys(),
                                          autopct='%1.1f%%', 
                                          startangle=90,
                                          colors=colors[:len(emotion_counts)])
        
        # Customize text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title('Emotion Distribution in Music Segments', fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'music_emotion_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_timeline_emotion_intensity(self):
        """Create comprehensive timeline of emotion intensity"""
        plt.figure(figsize=(16, 6))
        
        # Music segment data
        music_times = [m['start_time'] for m in self.music_emotions]
        music_emotions = [m['music_emotion'] for m in self.music_emotions]
        music_intensities = [m['music_intensity'] for m in self.music_emotions]
        
        # Color mapping
        emotion_colors = {
            'peaceful': '#90EE90', 'sad': '#4682B4', 'happy': '#FFD700',
            'exciting': '#FF4500', 'dramatic': '#800080', 'suspenseful': '#FFA500',
            'angry': '#DC143C', 'horror': '#000000'
        }
        
        colors = [emotion_colors.get(emotion, '#808080') for emotion in music_emotions]
        
        # Create scatter plot
        plt.scatter(music_times, music_intensities, c=colors, s=60, alpha=0.7, 
                   edgecolors='black', linewidth=0.5)
        
        # Add connecting lines
        plt.plot(music_times, music_intensities, 'k-', alpha=0.3, linewidth=1)
        
        # Customize plot
        plt.xlabel('Segment Index (Seconds)', fontsize=12)
        plt.ylabel('Emotion Intensity Score', fontsize=12)
        plt.title('Timeline of Emotion Intensity Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 10)
        
        # Create legend
        unique_emotions = list(set(music_emotions))
        legend_elements = [plt.scatter([], [], c=emotion_colors.get(emotion, '#808080'), 
                                     s=60, label=emotion) for emotion in unique_emotions]
        plt.legend(handles=legend_elements, title='Emotion Intensity', 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.graphs_dir, 'timeline_emotion_intensity.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_enhanced_visualizations(self):
        """Create all enhanced visualizations including new ones"""
        print("\U0001F4C8 Creating enhanced visualizations...")

        # Create original visualizations
        self.create_visualizations()

        # Create new dialogue and music visualizations
        if hasattr(self, 'dialogue_emotions') and self.dialogue_emotions:
            self.create_dialogue_emotion_timeline()

        if hasattr(self, 'music_emotions') and self.music_emotions:
            self.create_music_emotion_distribution()
            self.create_timeline_emotion_intensity()

        self.create_valence_arousal_trajectory()  # 🔥 Add this line

        print("✅ All enhanced visualizations created")

    def generate_html_report(self):
        print("📝 Generating HTML dashboard report...")

        # Load overall_analysis if not loaded
        if not self.overall_analysis:
            analysis_path = os.path.join(self.output_dir, "overall_analysis.xlsx")
            if os.path.exists(analysis_path):
                analysis_df = pd.read_excel(analysis_path)
                if not analysis_df.empty:
                    self.overall_analysis = analysis_df.iloc[0].to_dict()
                else:
                    print("⚠️ overall_analysis.xlsx is empty.")
            else:
                print("⚠️ overall_analysis.xlsx not found.")

        # Prepare visualizations
        visualizations = [
            {"title": "Comprehensive Dashboard", "path": os.path.join("visualizations", "comprehensive_dashboard.png")},
            {"title": "Emotion Timeline", "path": os.path.join("visualizations", "emotion_timeline.png")},
            {"title": "Emotion Distributions", "path": os.path.join("visualizations", "emotion_distributions.png")},
            {"title": "Person Analysis", "path": os.path.join("visualizations", "person_analysis.png")},
            {"title": "Intensity Heatmap", "path": os.path.join("visualizations", "intensity_heatmap.png")},
            {"title": "Dialogue Emotion Timeline", "path": os.path.join("visualizations", "dialogue_emotion_timeline.png")},
            {"title": "Music Emotion Distribution", "path": os.path.join("visualizations", "music_emotion_distribution.png")},
            {"title": "Timeline Emotion Intensity", "path": os.path.join("visualizations", "timeline_emotion_intensity.png")},
        ]

        # Build the full HTML directly
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Video Emotion Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                img {{ max-width: 100%; height: auto; }}
                .section {{ margin-bottom: 40px; }}
                .summary-box {{ background-color: #f9f9f9; border-left: 5px solid #3498db; padding: 20px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🎬 Video Emotion Analysis Report</h1>
            <div class="summary-box">
                <h2>Summary</h2>
                <p>{self.overall_analysis.get('emotion_statement', 'No summary generated.')}</p>
                <p><strong>Duration:</strong> {self.overall_analysis.get('total_duration', 'N/A')}s</p>
                <p><strong>Total Shots:</strong> {self.overall_analysis.get('total_shots', 'N/A')}</p>
                <p><strong>Dominant Audio Emotion:</strong> {self.overall_analysis.get('dominant_audio_emotion', 'N/A')}</p>
                <p><strong>Dominant Visual Emotion:</strong> {self.overall_analysis.get('dominant_visual_emotion', 'N/A')}</p>
                <p><strong>Average Intensity:</strong> {self.overall_analysis.get('average_intensity', 'N/A')}</p>
                <p><strong>Total Persons Detected:</strong> {self.overall_analysis.get('total_persons_detected', 'N/A')}</p>
                <p><strong>Person Density:</strong> {float(self.overall_analysis.get('person_density', 0))*100:.1f}%</p>
            </div>
        """

        for viz in visualizations:
            html_content += f"""
            <div class="section">
                <h2>{viz['title']}</h2>
                <img src="{viz['path'].replace(os.sep, '/')}">
            </div>
            """

        html_content += "</body></html>"

        # Write to file
        html_path = os.path.join(self.output_dir, "video_emotion_dashboard.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ HTML report generated: {html_path}")


    def run_enhanced_analysis(self):
        """Run enhanced analysis with dialogue and music emotion analysis"""
        print("🚀 Starting Enhanced Video Analysis with Dialogue and Music...")
        
        try:
            # Core analysis steps
            self.detect_shots()
            self.extract_shot_keyframes()
            self.analyze_shot_audio_emotions()
            self.analyze_shot_visual_emotions()
            
            # Enhanced analysis
            self.analyze_dialogue_emotions()
            self.analyze_music_segments()
            
            self.run_clip_analysis()
            self.generate_comprehensive_analysis()
            
            # Generate enhanced outputs
            self.create_enhanced_visualizations()
            self.save_detailed_reports()
            self.generate_html_report()
            
            # Save additional reports
            if hasattr(self, 'dialogue_emotions'):
                dialogue_df = pd.DataFrame(self.dialogue_emotions)
                dialogue_df.to_excel(os.path.join(self.output_dir, "dialogue_emotions.xlsx"), index=False)
            
            if hasattr(self, 'music_emotions'):
                music_df = pd.DataFrame(self.music_emotions)
                music_df.to_excel(os.path.join(self.output_dir, "music_emotions.xlsx"), index=False)
            
            print("\n" + "="*60)
            print("🎉 ENHANCED ANALYSIS COMPLETE!")
            print("="*60)
            print("📋 Additional Generated Files:")
            print("   • dialogue_emotions.xlsx - Dialogue emotion analysis")
            print("   • music_emotions.xlsx - Music segment emotion analysis")
            print("   • dialogue_emotion_timeline.png - Dialogue emotion timeline")
            print("   • music_emotion_distribution.png - Music emotion distribution")
            print("   • timeline_emotion_intensity.png - Comprehensive emotion timeline")
            
            return self.overall_analysis
            
        except Exception as e:
            print(f"❌ Error during enhanced analysis: {str(e)}")
            raise
        
        finally:
            if hasattr(self, 'cap'):
                self.cap.release()
    
    def create_valence_arousal_trajectory(self):
        """Compute and plot valence-arousal trajectory using audio features and custom logic."""
        print("\U0001F4C9 Creating valence-arousal trajectory using audio features...")

        valence_points = []
        arousal_points = []
        time_points = []

        for shot in self.shot_audio_emotions:
            start = int(shot['start_time'])
            end = int(shot['end_time'])

            features = {
                'tempo': shot.get('tempo', 0),
                'rms_mean': shot.get('rms', 0),
                'spectral_centroid_mean': shot.get('spectral_centroid', 0),
                'zcr_mean': shot.get('zero_crossing_rate', 0),
                'low_high_ratio': shot.get('harmonic_ratio', 1)
            }

            # === Custom valence-arousal logic ===
            tempo_norm = min(1.0, features['tempo'] / 180.0)
            energy_norm = min(1.0, features['rms_mean'] * 20)
            brightness_norm = min(1.0, features['spectral_centroid_mean'] / 3000.0)

            valence = (
                0.4 * brightness_norm +
                0.4 * (1 / (1 + features['low_high_ratio'])) -
                0.2 * features['zcr_mean']
            )
            valence = max(0.1, min(0.9, valence))

            arousal = (
                0.4 * tempo_norm +
                0.4 * energy_norm +
                0.2 * features['zcr_mean']
            )
            arousal = max(0.1, min(0.9, arousal))

            for t in range(start, end + 1):
                time_points.append(t)
                valence_points.append(valence)
                arousal_points.append(arousal)

        # Plotting
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        import os

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(valence_points, arousal_points, c=time_points, cmap='viridis', s=40, alpha=0.8)
        plt.plot(valence_points, arousal_points, linestyle='-', color='gray', alpha=0.5)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Time (Seconds)')

        plt.title("Valence-Arousal Trajectory", fontsize=14)
        plt.xlabel("Valence (Pleasantness)")
        plt.ylabel("Arousal (Energy)")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)

        # Add quadrant labels
        plt.text(0.75, 0.9, "Happy", fontsize=12)
        plt.text(0.05, 0.9, "Angry", fontsize=12)
        plt.text(0.05, 0.2, "Sad", fontsize=12)
        plt.text(0.75, 0.2, "Calm", fontsize=12)

        plt.tight_layout()
        out_path = os.path.join(self.graphs_dir, "valence_arousal_trajectory.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {out_path}")



#     def run_full_analysis(self):
#         """Run complete video analysis pipeline"""
#         print("🚀 Starting Enhanced Video Scene Analysis...")
#         print(f"📹 Analyzing video: {self.video_path}")
#         print(f"📊 Duration: {self.duration:.2f} seconds")
#         print(f"🎬 FPS: {self.fps:.2f}")
        
#         try:
#             # Core analysis steps
#             self.detect_shots()
#             self.extract_shot_keyframes()
#             self.analyze_shot_audio_emotions()
#             self.analyze_shot_visual_emotions()
#             self.run_clip_analysis()
#             self.generate_comprehensive_analysis()
            
#             # Generate outputs
#             self.create_visualizations()
#             self.save_detailed_reports()
            
#             # Summary
#             print("\n" + "="*60)
#             print("🎉 ANALYSIS COMPLETE!")
#             print("="*60)
#             print(f"📁 Results saved to: {self.output_dir}")
#             print(f"🎬 Total shots analyzed: {len(self.shots)}")
#             print(f"🎵 Dominant audio emotion: {self.overall_analysis.get('dominant_audio_emotion', 'N/A')}")
#             print(f"👁️ Dominant visual emotion: {self.overall_analysis.get('dominant_visual_emotion', 'N/A')}")
#             print(f"📈 Average intensity: {self.overall_analysis.get('average_intensity', 0):.1f}/10")
#             print(f"👥 Total persons detected: {self.overall_analysis.get('total_persons_detected', 0)}")
#             print("\n📋 Generated Files:")
#             print("   • shots_analysis.xlsx - Shot boundaries and timing")
#             print("   • audio_emotions.xlsx - Audio emotion analysis")
#             print("   • visual_emotions.xlsx - Visual emotion analysis")
#             print("   • person_detection.xlsx - Person detection data")
#             print("   • clip_scene_analysis.xlsx - CLIP scene classification")
#             print("   • combined_analysis.xlsx - Complete combined analysis")
#             print("   • overall_analysis.xlsx - Summary statistics")
#             print("   • comprehensive_dashboard.png - Visual dashboard")
#             print("   • Multiple visualization charts in /visualizations/")
#             print("   • Shot keyframes in /shots/")
            
#             return self.overall_analysis
            
#         except Exception as e:
#             print(f"❌ Error during analysis: {str(e)}")
#             raise
        
#         finally:
#             if hasattr(self, 'cap'):
#                 self.cap.release()


# Usage example
# if __name__ == "__main__":
#     # Example usage
#     video_path = "Hotstar Specials： Mistry ｜ Ram Kapoor ｜ Mona Singh ｜ Official Trailer ｜ Streaming June 27.mp4"  # Replace with your video path
    
#     if os.path.exists(video_path):
#         analyzer = EnhancedVideoSceneAnalyzer(video_path)
#         results = analyzer.run_enhanced_analysis()
        
#         print("\n🎬 Analysis Results:")
#         print(results['emotion_statement'])
#     else:
#         print("❌ Video file not found. Please check the path.")
#         print("💡 Make sure to:")
#         print("   1. Replace 'your_video.mp4' with your actual video file path")
#         print("   2. Ensure the Google Cloud credentials JSON file is in the correct location")
#         print("   3. Install all required dependencies:")
#         print("      pip install opencv-python librosa moviepy matplotlib seaborn")
#         print("      pip install google-cloud-videointelligence google-cloud-vision")
#         print("      pip install sentence-transformers transformers")
#         print("      pip install pysrt soundfile scikit-learn")

# if __name__ == "__main__":
#     # 1️⃣ Provide your YouTube link:
#     youtube_url = 'https://www.youtube.com/watch?v=fnF1hTlCag0'

#     # 2️⃣ Download video and subtitle
#     yt_downloader = YouTubeDownloader(youtube_url, download_path='./downloads', ffmpeg_path=r'C:\Users\jsingh11\OneDrive - dentsu\Image and Video Analytics\images\video_emotion_analysis\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe')
#     video_path = yt_downloader.download_video_and_subtitles()

#     if video_path:
#         yt_downloader.fallback_download_transcript()

#         # 3️⃣ Run your main EnhancedVideoSceneAnalyzer pipeline
#         analyzer = EnhancedVideoSceneAnalyzer(video_path)
#         results = analyzer.run_enhanced_analysis()

#         print("\n🎬 Analysis Results:")
#         print(results['emotion_statement'])
#     else:
#         print("❌ Download failed. Cannot proceed with video analysis.")

# if __name__ == "__main__":
#     # 1️⃣ Download YouTube Video
#     youtube_url = 'https://www.youtube.com/watch?v=fnF1hTlCag0'  # <== Put your URL here
#     yt_downloader = YouTubeDownloader(youtube_url, download_path='./downloads', ffmpeg_path=r'C:\Users\jsingh11\OneDrive - dentsu\Image and Video Analytics\images\video_emotion_analysis\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe')
#     video_path = yt_downloader.download_video_and_subtitles()

#     if video_path:
#         yt_downloader.fallback_download_transcript()

#         # 2️⃣ Run Emotion Analyzer (you already have this implemented)
#         # from video_emotion_analysis_module import EnhancedVideoSceneAnalyzer

#         analyzer = EnhancedVideoSceneAnalyzer(video_path)
#         results = analyzer.run_enhanced_analysis()
#         print("\n🎬 Video Emotion Analysis Completed")

#         # 3️⃣ Run LLM Auto Insights
#         combined_file = os.path.join(analyzer.output_dir, "combined_analysis.xlsx")
#         insight_generator = InsightGenerator(combined_file)
#         final_report = insight_generator.generate_insight()

#         with open(os.path.join(analyzer.output_dir, "auto_insights.txt"), "w", encoding="utf-8") as f:
#             f.write(final_report)
#         print("\n🎯 Auto Insights Generated:\n")
#         print(final_report)

#     else:
#         print("Download failed.")

# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------

st.set_page_config(page_title="🎬 Full Video Emotion Analyzer", layout="wide")
st.title("🎬 Complete Video Emotion Analyzer App")

# Input YouTube URL
youtube_url = st.text_input("Enter YouTube Video Link:")

# Optional override for FFmpeg path (Windows safe)
def get_ffmpeg_path():
    return r'C:\Users\jsingh11\OneDrive - dentsu\Image and Video Analytics\images\video_emotion_analysis\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe'

if st.button("🚀 Start Full Analysis"):
    if not youtube_url:
        st.warning("Please enter a valid YouTube link.")
    else:
        with st.spinner("Downloading Video & Subtitles..."):
            yt_downloader = YouTubeDownloader(
                youtube_url, download_path='./downloads', ffmpeg_path=get_ffmpeg_path())
            video_path = yt_downloader.download_video_and_subtitles()
            yt_downloader.fallback_download_transcript()

        if video_path:
            st.video(video_path)

            with st.spinner("Running Full Emotion Analysis..."):
                analyzer = EnhancedVideoSceneAnalyzer(video_path)
                analyzer.run_enhanced_analysis()

            st.success("✅ Analysis Complete!")

            combined_path = os.path.join(analyzer.output_dir, 'combined_analysis.xlsx')
            st.subheader("📊 Combined Analysis Data")
            df = pd.read_excel(combined_path)
            st.dataframe(df)

            st.subheader("📊 Visualizations")
            viz_dir = os.path.join(analyzer.output_dir, "visualizations")
            for img_file in os.listdir(viz_dir):
                if img_file.endswith(".png"):
                    st.image(os.path.join(viz_dir, img_file), caption=img_file, use_column_width=True)

            with st.spinner("Generating AI Insights via LLM..."):
                insight_gen = InsightGenerator(combined_path)
                insight = insight_gen.generate_insight()

            st.subheader("📄 AI-Generated Insight Report")
            st.markdown(insight)

            st.subheader("📄 Full HTML Dashboard")
            html_path = os.path.join(analyzer.output_dir, "video_emotion_dashboard.html")
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=2000, scrolling=True)

        else:
            st.error("❌ Video download failed.")
