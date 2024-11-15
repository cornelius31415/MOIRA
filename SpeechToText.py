import speech_recognition as sr

# Zeige alle verfügbaren Mikrofone an
print("Available microphones:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{index}: {name}")

# Mikrofon auswählen (z.B. das Mikrofon mit Index 1)
mic_index = 0  # Index des gewünschten Mikrofons einfügen
recognizer = sr.Recognizer()

# Funktion für kontinuierliches Hören und Erkennen von Sprache mit ausgewähltem Mikrofon
def continuous_listen():
    with sr.Microphone(device_index=mic_index) as source:
        print("Assistant is ready. Press Ctrl+C to stop.")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        recognizer.pause_threshold = 0.5

        while True:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                print("Recognizing speech...")
                text = recognizer.recognize_google(audio, language="en-US")
                print(f"You said: {text}")
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
            except sr.RequestError:
                print("Request error; please check your internet connection.")

# Programm starten
try:
    continuous_listen()
except KeyboardInterrupt:
    print("\nProgram terminated.")
