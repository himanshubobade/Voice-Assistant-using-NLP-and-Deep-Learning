
import pyttsx3 #pip install pyttsx3
import speech_recognition as sr #pip install speechRecognition
import datetime
import wikipedia #pip install wikipedia
import webbrowser
import os
import smtplib

##########################
import cv2,time #opencv
import random
import spacy
##########################

##########################
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.stem import WordNetLemmatizer
#nltk.download('popular', quiet=True)
#nltk.download('punkt')
#nltk.download('wordnet')
##########################

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[1].id)


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        print("Good Morning!")
        speak("Good Morning!")

    elif hour>=12 and hour<18:
        print("Good Afternoon!")
        speak("Good Afternoon!")   

    else:
        print("Good Evening!")
        speak("Good Evening!")  

    print("I am your voice assistant lynda Sir. Please tell me how may I help you")
    speak("I am your voice assistant lynda Sir. Please tell me how may I help you")       

def takeCommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        print(e)    
        print("Say that again please...")  
        return "None"
    return query

def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login('youremail@email.com', '#######')
    server.sendmail('youremail@email.com', to, content)
    server.close()

def photo1(face_cascade,smile_cascade):
    #photo1(face_cascade,smile_cascade)
    number = random.randint(0,9)
    number = str(number)+".jpg"
    photo = number
    original_image = cv2.imread(photo)
    if original_image is not None:
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=4)
        #profiles_not_faces = [x for x in detected_smiles if x not in detected_faces]
        #print(profiles_not_faces)
        for (x, y, width, height) in detected_faces:
            cv2.rectangle(original_image,(x, y),(x + width, y + height),(0,255,0),thickness=2)
            #detected_smiles = smile_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=4)
            smile=smile_cascade.detectMultiScale(image,scaleFactor=1.8,minNeighbors=50)
            for x1,y1,w,h in smile:
                img=cv2.rectangle(original_image,(x1,y1),(x1+w,y1+h),(255,0,0),thickness=10)

        cv2.imshow(f'Detected Faces in {photo}', original_image)
        print(f'Detected Faces in {photo}')
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    else:
        print(f'En error occurred while trying to load {photo}')

def cam_smile(face_cascade,smile_cascade):
    video=cv2.VideoCapture(0)

    while True:
        check,frame=video.read()
        #print(check)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        #print(face)
        for x,y,w,h in face:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            smile=smile_cascade.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=20)
            for x,y,w,h in smile:
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),10)

        cv2.imshow('gotcha',frame)
        key=cv2.waitKey(1)

        if key==ord('q'):
            break

    video.release()
    cv2.destroyAllWindows

def beta():
    with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
        raw = fin.read().lower()

    #TOkenisation
    sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
    word_tokens = nltk.word_tokenize(raw)# converts to list of words

    # Preprocessing
    lemmer = WordNetLemmatizer()
    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


    # Keyword Matching
    GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
    GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

    def greeting(sentence):
        """If user's input is a greeting, return a greeting response"""
        for word in sentence.split():
            if word.lower() in GREETING_INPUTS:
                return random.choice(GREETING_RESPONSES)


    # Generating response
    def response(user_response):
        robo_response=''
        sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx=vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if(req_tfidf==0):
            robo_response=robo_response+"I am sorry! I don't understand you"
            return robo_response
        else:
            robo_response = robo_response+sent_tokens[idx]
            return robo_response


    flag=True
    print("My name is lynda. I will answer your queries. If you want to exit, say Bye!")
    speak("My name is lynda. I will answer your queries. If you want to exit, say Bye!")
    while(flag==True):
        user_response = takeCommand().lower()
        user_response=user_response.lower()
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                print("You are welcome..")
                speak("You are welcome..")
            else:
                if(greeting(user_response)!=None):
                    print(greeting(user_response))
                    speak(greeting(user_response))
                else:
                    #print(response(user_response))
                    print(response(user_response))
                    speak(response(user_response))
                    sent_tokens.remove(user_response)
        else:
            flag=False
            print("Bye! take care.. Closing the beta mode")
            speak("Bye! take care.. Closing the beta mode")
    

if __name__ == "__main__":
    wishMe()
    while True:
    # if 1:
        query = takeCommand().lower()

        # Logic for executing tasks based on query
        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia")
            print(results)
            speak(results)

        elif 'google search' in query:
            speak("What should i search for you?")
            query = takeCommand().lower()
            #search = record_audio('What do you want to search for?')
            url = 'https://google.com/search?q=' + query
            webbrowser.get().open(url)
            print('Here is what i found for '+ query)
            speak('Here is what i found for '+ query)

        elif 'your name' in query: 
            print('I am lynda')
            speak('I am lynda')

        elif 'open youtube' in query:
            webbrowser.open("youtube.com")

        elif 'location' in query:
            print('What is the location  you want to search?')
            speak('What is the location  you want to search?')
            query = takeCommand().lower()
            url = 'https://google.nl/maps/place/' + query + '/&amp;'
            webbrowser.get().open(url)
            print('Here is your location '+ query)
            speak('Here is your location '+ query)
        

        elif "smart" in query:
            print("Thanks! But you are smarter,so help me out to be smarter")
            speak("Thanks! But you are smarter,so help me out to be smarter") 


        elif 'play music' in query:
            music_dir = 'C:\\Users\\HIMANSHUU\\Desktop\\Voice assistant\\exp\\music'
            songs = os.listdir(music_dir)
            print(songs)    
            os.startfile(os.path.join(music_dir, songs[0]))

        elif 'document' in query:
            doc_dir = "C:\\Users\\HIMANSHUU\\Desktop\\Voice assistant\\exp\\docs"
            docs= os.listdir(doc_dir)
            print(docs)
            os.startfile(os.path.join(doc_dir,docs[0]))

        elif 'time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")    
            print(f"Sir, the time is {strTime}")
            speak(f"Sir, the time is {strTime}")

        elif 'spotify' in query:
            codePath = "C:\\Users\\HIMANSHUU\\AppData\\Roaming\\Spotify\\Spotify.exe"
            os.startfile(codePath)

        elif 'email' in query:
            try:
                print("What should I say?")
                speak("What should I say?")
                content = takeCommand()
                to = "receiveremail@email.com"    
                sendEmail(to, content)
                print("Email has been sent!")
                speak("Email has been sent!")
            except Exception as e:
                #print(e)
                print("Sorry Sir. I am not able to send this email")
                speak("Sorry Sir. I am not able to send this email")  

        elif "exit" in query:
            print("Thank you! It was a pleasure serving you")
            speak("Thank you! It was a pleasure serving you")
            exit()

        elif "deep learning" in query:
            print("Processing data sir")
            speak("Processing data sir")
            try:
                print("Heres the output")
                speak("Heres the output")
                face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                smile_cascade=cv2.CascadeClassifier("smile.xml")
                photo1(face_cascade,smile_cascade)
            except:
                print("Sorry sir an error occured")
                speak("Sorry sir an error occured")

        elif "camera" in query:
            print("setting up your camera")
            speak("setting up your camera")
            try:
                face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                smile_cascade=cv2.CascadeClassifier("smile.xml")
                cam_smile(face_cascade,smile_cascade)

            except:
                print("Sorry sir an error occured")
                speak("Sorry sir an error occured")

        elif "words" in query:
            print("tell me first word to compare")
            speak("tell me first word to compare")
            query = takeCommand().lower()
            word1 = query
            print("tell me second word to compare")
            speak("tell me second word to compare")
            query = takeCommand().lower()
            word2 = query
            try:
                print("Processing data sir")
                speak("Processing data sir")
                nlp = spacy.load('en_core_web_md')
                token1 = nlp(word1)
                token2 = nlp(word2)
                string = "The word "+str(token1.text)+" and "+str(token2.text)+" have a similarity score of "+ str(round(abs(token1.similarity(token2))*100,2))+" percent "
                print(string)
                speak(string)

            except:
                print("An error loading NLP modules")
                speak("An error loading NLP modules")  

        elif "beta" in query:
            print("Welcome to beta mode of our project. This mode is still in construction so the output data still might be inconsistant. ")
            speak("Welcome to beta mode of our project. This mode is still in construction so the output data still might be inconsistant. ")
            beta()

        else:
            print("Sorry, I didn't get you")
            speak("Sorry, I didn't get you")   

