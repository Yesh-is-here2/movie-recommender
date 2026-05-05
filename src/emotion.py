# emotion.py
# Handles facial emotion detection using the DeepFace library.
# When a user clicks SelfieSearch, their webcam image is sent here
# as a base64 string. We decode it, analyze the face, and return
# the detected emotion so we can recommend matching movies.

import base64
import numpy as np
import os


def decode_base64_image(base64_str: str):
    """
    Convert a base64-encoded image string (from the browser webcam)
    into a NumPy array that OpenCV and DeepFace can process.

    The browser sends images as data URLs like:
    'data:image/jpeg;base64,/9j/4AAQ...'
    We strip the header part and decode just the raw base64 data.
    """
    import cv2

    # Remove the data URL header if present (e.g. 'data:image/jpeg;base64,')
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    # Decode base64 string to raw bytes
    img_bytes = base64.b64decode(base64_str)

    # Convert bytes to a NumPy array and decode as an image
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def analyze_emotion(base64_image: str) -> str:
    """
    Analyze the dominant facial emotion from a base64 webcam image.
    Returns a string like 'happy', 'sad', 'angry', 'fear', etc.

    We try multiple face detector backends in order (opencv → retinaface → mtcnn)
    because some backends work better depending on lighting and face angle.
    If all backends fail, we default to 'happy' so the user still gets results.

    One key design decision: if 'neutral' wins but its score is very high
    and other emotions exist, we override it with the next strongest emotion.
    This gives users more interesting and relevant movie recommendations
    rather than always defaulting to neutral/drama picks.
    """
    try:
        from deepface import DeepFace
        import cv2

        # Decode the incoming base64 image
        img = decode_base64_image(base64_image)
        if img is None:
            return "neutral"

        # Resize to a standard resolution for consistent detection
        img = cv2.resize(img, (640, 480))

        # Try multiple face detector backends — opencv is fastest,
        # retinaface and mtcnn are more accurate but slower
        backends = ["opencv", "retinaface", "mtcnn"]
        result = None

        for backend in backends:
            try:
                result = DeepFace.analyze(
                    img,
                    actions=["emotion"],       # Only analyze emotion, skip age/gender/race
                    enforce_detection=False,   # Don't crash if no face is detected
                    detector_backend=backend,
                    silent=True                # Suppress DeepFace's internal print logs
                )
                print(f"✅ Detection succeeded with backend: {backend}")
                break  # Stop trying backends once one succeeds
            except Exception as e:
                print(f"⚠️ Backend {backend} failed: {e}")
                continue

        # If all backends failed, default to happy
        if result is None:
            return "happy"

        # DeepFace returns a list when multiple faces are detected
        # We only care about the first (most prominent) face
        if isinstance(result, list):
            result = result[0]

        # Get the dictionary of emotion → confidence score (percentage)
        emotions = result.get("emotion", {})
        print(f"🎭 Emotion scores: {emotions}")

        if not emotions:
            return "happy"

        # Sort emotions from highest to lowest confidence score
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        print(f"🎭 Sorted: {sorted_emotions}")

        top_emotion, top_score = sorted_emotions[0]

        # Neutral tends to dominate even when the user has a clear expression.
        # If neutral wins, we look at the next strongest non-neutral emotion
        # and use that instead — this gives better movie recommendations.
        # DeepFace was returning neutral almost every time during testing
        # so we added this override to pick the next strongest emotion instead
        if top_emotion == "neutral":
            non_neutral = [(e, s) for e, s in sorted_emotions if e != "neutral"]
            if non_neutral:
                best_non_neutral, score = non_neutral[0]
                print(f"🎭 Neutral overridden → {best_non_neutral} ({score:.1f}%)")
                return best_non_neutral.lower()
            return "happy"  # Fallback if truly no other emotion detected

        print(f"🎭 Winner: {top_emotion} = {top_score:.1f}%")
        return top_emotion.lower()

    except Exception as e:
        print(f"❌ Emotion detection error: {e}")
        return "happy"  # Always return something valid so the app doesn't break


def emotion_to_message(emotion: str) -> str:
    """
    Return a friendly human-readable message based on the detected emotion.
    This message is displayed to the user on the dashboard above their
    mood-based movie recommendations.
    """
    messages = {
        "happy":    "You look happy! Here are some exciting picks for you 🎉",
        "sad":      "Feeling down? These feel-good movies will cheer you up 😊",
        "angry":    "Take a breather — some comedy might help 😄",
        "fear":     "Feeling anxious? Let's lighten the mood 🌟",
        "surprise": "Surprised? Here are some thrillers to match that energy 🎭",
        "disgust":  "Let's flip that mood with something fun 🎬",
        "neutral":  "Here are today's top picks for you 🍿",
    }
    # Return matching message or a generic fallback
    return messages.get(emotion, "Here are some movies you might enjoy 🎬")