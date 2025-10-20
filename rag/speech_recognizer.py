from speech_recorder import detect_question
import speech_recognition as sr

def transcribe_qestion():

    audio = detect_question()

    r = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio = r.listen(source)  # Read the entire audio file
    try:
        text = r.recognize_google(audio)
        print("Google Speech Recognition thinks you said: " + text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")