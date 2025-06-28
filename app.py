# app.py
from flask import Flask, request, jsonify
from transcript import give_summary
from flask_cors import CORS
import threading
from concentration_tracker import track_concentration
from flask import Response
import time
from web_summarizer import summarize_web
from emotion_detector import emotion_detector
import numpy as np


app = Flask(__name__)
CORS(app)

# Start concentration tracker in a background thread
concentration_triggered = False
score_container = {'value': 0}
frame_container = {'frame': None}
distraction_container = {'value': 0}

def monitor_concentration():
    global concentration_triggered
    concentration_triggered = track_concentration(score_container,frame_container,distraction_container)

def detect_emotion_from_audio(audio_frame):
    """Real emotion detection from audio frame using trained model"""
    return emotion_detector.detect_emotion(audio_frame)

@app.route('/api/audio-emotion-stream', methods=['POST'])
def audio_emotion_stream():
    """Real-time audio emotion analysis with distraction/concentration flags"""
    try:
        print("Received audio emotion request")
        # Get audio frame data from request
        data = request.get_json()
        audio_frame = data.get('audio_frame', None)
        
        if audio_frame is None:
            print("No audio frame provided")
            return jsonify({'error': 'No audio frame provided'}), 400
        
        print(f"Processing audio frame with {len(audio_frame.get('data', []))} samples")
        
        # Handle different audio data formats from frontend
        if isinstance(audio_frame, dict):
            # Frontend sends frequency data or raw audio data
            if 'data' in audio_frame:
                # Convert frequency data to a format the emotion detector can process
                audio_data = audio_frame['data']
                if isinstance(audio_data, list) and len(audio_data) > 0:
                    # Convert to numpy array for processing
                    audio_array = np.array(audio_data, dtype=np.float32)
                    
                    # Quick normalization for real-time processing
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array / max_val
                    
                    print(f"Processing audio array with shape: {audio_array.shape}")
                    # Process through emotion detection
                    emotion_result = detect_emotion_from_audio(audio_array)
                else:
                    print("Empty audio data, using fallback")
                    # Fallback for empty data
                    emotion_result = detect_emotion_from_audio({'data': 'empty'})
            else:
                # Direct audio array
                emotion_result = detect_emotion_from_audio(audio_frame)
        else:
            # Direct audio array
            emotion_result = detect_emotion_from_audio(audio_frame)
        
        print(f"Emotion result: {emotion_result}")
        # Return the emotion analysis with distraction/concentration flags
        return jsonify(emotion_result)
        
    except Exception as e:
        print(f"Error in audio emotion stream: {e}")
        # Return a fallback result instead of error for real-time continuity
        return jsonify({
            "emotion": "calm",
            "confidence": 0.7,
            "distracted": False,
            "concentration": 0.6,
            "probabilities": {
                "focused": 0.2, "distracted": 0.1, "engaged": 0.2,
                "bored": 0.1, "excited": 0.1, "calm": 0.3
            }
        })
    
@app.route('/video_feed')
def video_feed():
    """Video streaming route that reads from frame_container."""
    def generate():
        while True:
            if frame_container['frame'] is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_container['frame'] + b'\r\n')
            time.sleep(0.033)  
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/web_summarize', methods=['POST'])
def web_summarize():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        link = data.get('link')
        
        if not link:
            return jsonify({'error': 'No link provided'}), 400

        print(f"Received web link: {link}")  # Debug log
        
        summary = summarize_web(link)
        
        if summary:
            return jsonify({'summary': summary, 'status': 'success'})
        else:
            return jsonify({'error': 'Failed to summarize the web page'}), 500
            
    except Exception as e:
        print(f"Error in web_summarize: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/score_feed')
def score_feed():
    def event_stream():
        while True:
            score = score_container['value']
            yield f"data: {score}\n\n"
            time.sleep(0.5)

    return Response(event_stream(), mimetype='text/event-stream')

@app.route('/distracted_feed')
def distracted_feed():
    def event_stream():
        while True:
            score = distraction_container['value']
            yield f"data: {score}\n\n"
            time.sleep(0.5)

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/summarize', methods=['POST'])
def summarize():
    print('starting summarize endpoint')
    global concentration_triggered

    data = request.get_json()
    link = data.get('link')
    time_stamp = data.get('time_stamp', 0)
    print(f"Received link: {link}, time_stamp: {time_stamp}")

    # Start concentration monitoring
    concentration_thread = threading.Thread(target=monitor_concentration)
    concentration_thread.start()
    concentration_thread.join()

    if concentration_triggered:
        summarizer = give_summary(link)
        summary, quiz = summarizer.summarize(time_stamp)
        return jsonify({'summary': summary, 'quiz': quiz})
    else:
        return jsonify({'summary': '', 'quiz': [], 'message': 'User did not lose concentration'})

if __name__ == "__main__":
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
