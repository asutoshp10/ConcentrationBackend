from youtube_transcript_api import YouTubeTranscriptApi
import json
import re
import os
from openai import OpenAI  # Using Nebius AI's OpenAI-compatible API

# Initialize Nebius AI client
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDEwOTExMTA1NjI5MTk5MDAwNTk1NiIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwODgzMDg3NSwidXVpZCI6ImU3MWQ2OTM0LTUxMTEtNDAzNC1iMjIzLTk5NTFiZGZkMzQ5MiIsIm5hbWUiOiJmYXN0IiwiZXhwaXJlc19hdCI6IjIwMzAtMDYtMjdUMjI6NDc6NTUrMDAwMCJ9.-AYoiH3tcZpxW6SFQ1mKPelAYVUPim52LfOnMdQj7FE"
)

class give_summary:
    def __init__(self, link):
        self.link = link
        neg = 'https://www.youtube.com/watch?v='
        if link.startswith(neg):
            self.pos = link[len(neg):]
            self.pos = self.pos.split('&')[0]  # Clean parameters
        else:
            self.pos = None
        print(f"Video ID: {self.pos}")
    
    def summarize(self, time_stamp=0):
        self.real_trasn = ''
        video_id = self.pos
        if not video_id:
            return "not a valid yt link", []
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            print(f'Transcript Exception: {e}')
            return ("Sorry, I couldn't retrieve the transcript for this video. This might be because:\n- The video doesn't have captions\n- The video is private or unavailable\n- There was a temporary error\n\nPlease try with a different video that has available captions.", [])
        
        if not transcript:
            return "Sorry, I couldn't retrieve the transcript. Please try a different video.", []
            
        if time_stamp == 0:
            self.real_trasn = " ".join([t['text'] for t in transcript])
        else:
            self.real_trasn = " ".join([t['text'] for t in transcript if t['start'] <= time_stamp])

        if not self.real_trasn.strip():
            return "The transcript is empty. Please try a different video.", []

        # Generate summary using Nebius AI
        try:
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes YouTube transcripts in detail."},
                    {"role": "user", "content": f"Summarize this transcript:\n\n{self.real_trasn}"}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            summary_text = response.choices[0].message.content
        except Exception as e:
            print(f"Summary Error: {e}")
            summary_text = "Summary generation failed. Please try again."
    
        # Generate quiz using Nebius AI
        quiz = []
        try:
            quiz_response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-fast",
                messages=[
                    {"role": "system", "content": "Generate 3 multiple-choice questions with 4 options each in JSON format. Use format: {'questions': [{'question':'...','options':['A','B','C','D'],'answer':'B'}]}"},
                    {"role": "user", "content": f"Create quiz questions from:\n\n{self.real_trasn}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1024
            )
            quiz_json = quiz_response.choices[0].message.content
            quiz_data = json.loads(quiz_json)
            quiz = quiz_data.get('questions', [])
            if not isinstance(quiz, list):
                quiz = []
            print("Quiz JSON:", quiz_json)
        except json.JSONDecodeError:
            print("Quiz JSON decode error. Trying regex fallback...")
            match = re.search(r"\[.*\]", quiz_json, re.DOTALL)
            if match:
                try:
                    quiz = json.loads(match.group(0))
                except:
                    quiz = []
        except Exception as e:
            print(f"Quiz Error: {e}")
            quiz = []
        
        return summary_text, quiz

# Example usage
# s = give_summary('https://www.youtube.com/watch?v=lg48Bi9DA54')
# a, q = s.summarize()
# # print(f'Summary: {a}')
# print(f'Quiz: {q}')
