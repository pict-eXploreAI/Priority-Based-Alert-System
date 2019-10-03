
from googletrans import Translator
import os
# import espeaksss
from gtts import gTTS 
from pygame import mixer
from tempfile import TemporaryFile

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
        # tts.save("welcome.mp3")
        # os.system("mpg321 welcome.mp3")
        # print("Done!")
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