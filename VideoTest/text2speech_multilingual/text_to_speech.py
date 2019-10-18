from googletrans import Translator
import os
from gtts import gTTS 
from pygame import mixer
from tempfile import TemporaryFile

# Standard Text-To-Speech code using Google's Text-To-Speech model
def give_message(message, language="English"):
    text = message
    destination_language = {
        "English":"en",
        "Hindi":"hi",
        "Marathi":"mr",
        "Malyalam":"ml",
        "Punjabi":"pa",
        "Kanada":"ka",
        "Telugu":"te",
        "Tamil":"ta",
    }
    import os
    translator=Translator()
    try:
        key = language
        value = destination_language[language]
        text1=translator.translate(text, dest=value).text
        print(key + translator.translate(text, dest=value).text)
        print(text1)
        tts = gTTS(text=text1, lang=value, slow=False)
        mixer.init()
        sf = TemporaryFile()
        tts.write_to_fp(sf)
        sf.seek(0)
        mixer.music.load(sf)
        mixer.music.play()
    except:
        print("Wrong input")


if __name__ == "__main__":
    give_message("Car is on your left, you are going to hit the wall. Please walk on road.", "English")
