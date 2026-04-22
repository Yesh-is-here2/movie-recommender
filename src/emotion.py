import base64
import numpy as np
import os

def decode_base64_image(base64_str: str):
    import cv2
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def analyze_emotion(base64_image: str) -> str:
    try:
        from deepface import DeepFace
        import cv2

        img = decode_base64_image(base64_image)
        if img is None:
            return "neutral"

        img = cv2.resize(img, (640, 480))

        backends = ["opencv", "retinaface", "mtcnn"]
        result = None

        for backend in backends:
            try:
                result = DeepFace.analyze(
                    img,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend=backend,
                    silent=True
                )
                print(f"✅ Detection succeeded with backend: {backend}")
                break
            except Exception as e:
                print(f"⚠️ Backend {backend} failed: {e}")
                continue

        if result is None:
            return "happy"

        if isinstance(result, list):
            result = result[0]

        emotions = result.get("emotion", {})
        print(f"🎭 Emotion scores: {emotions}")

        if not emotions:
            return "happy"

        # Sort all emotions by score
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        print(f"🎭 Sorted: {sorted_emotions}")

        top_emotion, top_score = sorted_emotions[0]

        # If neutral is winning, pick the strongest non-neutral emotion instead
        if top_emotion == "neutral":
            non_neutral = [(e, s) for e, s in sorted_emotions if e != "neutral"]
            if non_neutral:
                best_non_neutral, score = non_neutral[0]
                print(f"🎭 Neutral overridden → {best_non_neutral} ({score:.1f}%)")
                return best_non_neutral.lower()
            return "happy"

        print(f"🎭 Winner: {top_emotion} = {top_score:.1f}%")
        return top_emotion.lower()

    except Exception as e:
        print(f"❌ Emotion detection error: {e}")
        return "happy"

def emotion_to_message(emotion: str) -> str:
    messages = {
        "happy":    "You look happy! Here are some exciting picks for you 🎉",
        "sad":      "Feeling down? These feel-good movies will cheer you up 😊",
        "angry":    "Take a breather — some comedy might help 😄",
        "fear":     "Feeling anxious? Let's lighten the mood 🌟",
        "surprise": "Surprised? Here are some thrillers to match that energy 🎭",
        "disgust":  "Let's flip that mood with something fun 🎬",
        "neutral":  "Here are today's top picks for you 🍿",
    }
    return messages.get(emotion, "Here are some movies you might enjoy 🎬")