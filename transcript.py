from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import json
import re

genai.configure(api_key="AIzaSyCt07QcDpiIcmMafQ8EzC1U0fhF2apfZ8o")
model = genai.GenerativeModel("models/gemini-1.5-flash")

class give_summary():
    def __init__(self,link):
        self.link=link
        neg='https://www.youtube.com/watch?v='
        if link.startswith(neg):
            self.pos=link[len(neg):]
            # Clean up the video ID by removing any additional parameters
            self.pos = self.pos.split('&')[0]  # Remove playlist and other parameters
        else:
            return 'not a valid yt link'
        print(f"Video ID: {self.pos}")
    
    def summarize(self,time_stamp=0):
        self.real_trasn=''
        video_id=self.pos
        # No need to split again since we already cleaned it in __init__
        transcript = None
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            print(f'Transcript Exception: {e}')
            # Return fallback response when transcript fails
            return "Sorry, I couldn't retrieve the transcript for this video. This might be because:\n- The video doesn't have captions\n- The video is private or unavailable\n- There was a temporary error\n\nPlease try with a different video that has available captions.", []
        
        if transcript is None:
            return "Sorry, I couldn't retrieve the transcript for this video. Please try with a different video that has available captions.", []
            
        if time_stamp==0:
            self.real_trasn=" ".join([t['text'] for t in transcript])
        else:
            self.real_trasn=" ".join([t['text'] for t in transcript if t['start']<=time_stamp])

        if not self.real_trasn.strip():
            return "The transcript is empty or couldn't be processed. Please try with a different video.", []

        prompt_s = f"Summarize this transcript in detail:\n\n{self.real_trasn}"
        try:
            response = model.generate_content(prompt_s)
            summary_text = response.text
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary_text = "Sorry, I couldn't generate a summary due to an error. Please try again."
    
        prompt_q = f"""
        Generate 3 multiple choice questions from the following transcript:
        - Each question should have 4 options
        - Return the result in JSON with format:
        [{{"question": "...", "options": ["A", "B", "C", "D"], "answer": "B"}}, ...]

        Transcript:
        {self.real_trasn}
        """
        try:
            quiz_output = model.generate_content(prompt_q) 
            try:
                match = re.search(r"\[.*\]", quiz_output.text, re.DOTALL)
                if match:
                    quiz = json.loads(match.group(0))
                else:
                    quiz = []
            except json.JSONDecodeError:
                print('Error parsing quiz JSON')
                quiz = []
        except Exception as e:
            print(f"Error generating quiz: {e}")
            quiz = []
            
        return summary_text, quiz


