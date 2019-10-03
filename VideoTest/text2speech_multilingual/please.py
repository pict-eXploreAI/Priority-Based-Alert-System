from googletrans import Translator
import os
import espeak
from gtts import gTTS 
text=("You are currently walking on the road")
destination_language = {
    "English":"en",
    "Hindi":"hi",
    "Marathi":"mr",
    "Malyalam":"ml",
    "Punjabi":"pa",
    "Kanada":"ka",
    "Telugu":"te",
    "Tamil":"ta"
}
es = espeak.ESpeak()
translator=Translator()
for key, value in destination_language.items():
    text1=translator.translate(text, dest=value).text
    print(translator.translate(text, dest=value).text)
    es.say(text1)
    myobj = gTTS(text=text1, lang=value, slow=False) 
    myobj.save("welcome"+value+".mp3") 
    #y, sr = librosa.load(es.say(text1),duration=5.0)
    #librosa.output.write_wav('file_trim_5s.wav', y, sr)
    #os.system("espeak translator.translate(text, dest=value).text")
