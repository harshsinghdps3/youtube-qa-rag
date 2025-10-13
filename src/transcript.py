import os
import json
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from pathlib import Path

class TranscriptRetriever:
    def __init__(self, cache_dir="./data/transcripts"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.whisper_model = None
        self.ytt_api = YouTubeTranscriptApi()  # Initialize the API instance
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        if "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        elif "watch?v=" in url:
            return url.split("watch?v=")[1].split("&")[0]
        raise ValueError("Invalid YouTube URL")
    
    def get_transcript(self, video_url: str) -> list[dict]:
        """Retrieve transcript with time codes; fall back to Whisper if needed."""
        video_id = self.extract_video_id(video_url)
        cache_path = self.cache_dir / f"{video_id}.json"
        
        # Check cache
        if cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f)
        
        # Try YouTube API first
        try:
            # Use the new API: fetch() returns a FetchedTranscript object
            fetched_transcript = self.ytt_api.fetch(video_id)
            
            # Convert to raw data format
            transcript_data = fetched_transcript.to_raw_data()
            
            segments = [
                {"start": seg["start"], "duration": seg["duration"], "text": seg["text"]}
                for seg in transcript_data
            ]
            
            with open(cache_path, "w") as f:
                json.dump(segments, f, indent=2)
            
            return segments
            
        except (TranscriptsDisabled, NoTranscriptFound, Exception) as e:
            print(f"Captions unavailable for {video_id} ({type(e).__name__}), falling back to Whisper...")
            return self._whisper_fallback(video_url, video_id, cache_path)
    
    def _whisper_fallback(self, video_url: str, video_id: str, cache_path: Path) -> list[dict]:
        """Download audio and transcribe with Whisper."""
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model("base")
        
        audio_path = self.cache_dir / f"{video_id}.mp3"
        
        # Download audio with error handling
        result = os.system(f"yt-dlp -x --audio-format mp3 -o {audio_path} {video_url}")
        if result != 0 or not audio_path.exists():
            raise RuntimeError(f"Failed to download audio from {video_url}")
        
        try:
            result = self.whisper_model.transcribe(str(audio_path), word_timestamps=True)
            segments = [
                {"start": seg["start"], "duration": seg["end"] - seg["start"], "text": seg["text"]}
                for seg in result["segments"]
            ]
            
            with open(cache_path, "w") as f:
                json.dump(segments, f, indent=2)
            
            return segments
        finally:
            # Clean up audio file
            if audio_path.exists():
                audio_path.unlink()
