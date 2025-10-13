# tests/test_transcript.py or test_transcript.py
from src.transcript import TranscriptRetriever

def test_transcript():
    retriever = TranscriptRetriever(cache_dir=".data/transcripts")
    
    # Test with videos known to have captions
    test_videos = [
        "https://www.youtube.com/watch?v=_DemxyrqhY4"   
    ]
    
    for video_url in test_videos:
        print(f"\n{'='*60}")
        print(f"Testing: {video_url}")
        print(f"{'='*60}")
        
        try:
            segments = retriever.get_transcript(video_url)
            print(f"✓ Successfully retrieved {len(segments)} segments")
            print("\nFirst 3 segments:")
            for seg in segments[:3]:
                print(f"  [{seg['start']:.2f}s] {seg['text'][:60]}...")
        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_transcript()
