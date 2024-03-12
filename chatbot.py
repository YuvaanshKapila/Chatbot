import re
import nltk
from nltk.chat.util import Chat, reflections
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.models import load_model

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Example training data
words = ["hello", "how", "are", "you", "today", "I", "am", "good", "bye"]
classes = ["greeting", "farewell", "mood", "identity", "humor"] 

documents = [
    (["hello", "hi", "hey"], "greeting"),
    (["bye", "goodbye", "see", "you"], "farewell"),
    (["how", "are", "you", "doing"], "mood"),
    (["what", "is", "your", "name"], "identity"),
    (["tell", "me", "a", "joke"], "humor"),
]


output_empty = [0] * len(classes)

# Create training data
training = []

for doc in documents:
    bag = [0] * len(words)

    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        if w in pattern_words:
            bag[words.index(w)] = 1

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Separate features and labels before converting to numpy array
train_x = [item[0] for item in training]
train_y = [item[1] for item in training]

# Convert lists to numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)



# Define the network model
model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

#'ERROR_THRESHOLD' defined
ERROR_THRESHOLD = 0.2

# Define some basic pairs for the Chat class
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, How are you today?",]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there", "Hi! How can I help you?", "Greetings!"]
    ],
    [
        r"quit",
        ["Bye. It was nice talking to you. See you soon :)"]
    ],
    [
        r"(.*) (joke|funny)",
        ["Why did the chatbot go to therapy? It had too many issues!", 
         "Sure, here's a joke: Why don't scientists trust atoms? Because they make up everything!",
         "Knock, knock. Who's there? Atch. Atch who? Bless you!"]
    ],
    # Additional pairs for general conversation
    [
        r"how are you",
        ["I'm just a computer program, but thanks for asking!", "I don't have feelings, but I'm here to assist you."]
    ],
    [
        r"what is your purpose",
        ["I'm here to help answer your questions and have a chat with you."]
    ],
    [
        r"(.*)",
        ["I'm not sure about that. Can you tell me more?", "I'm still learning. What do you mean by %1?"]
    ]
]


# This will hold the unknown questions and their answers
learned_data = {}

def preprocess_input(user_input):
    tokens = nltk.word_tokenize(user_input.lower())  # Tokenize and convert to lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]  # Remove stop words
    return ' '.join(filtered_tokens)

def bow(sentence, words, show_details=True):
    # implementation of the bag-of-words function here
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in nltk.word_tokenize(sentence)]
    bag = [1 if lemmatized_token in lemmatized_tokens else 0 for lemmatized_token in words]

    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)

    if p is None:
        print("Unable to generate bag of words for the input.")
        return []

    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)



    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list

if __name__ == "__main__":
    print("Hi, I'm a chatbot. You can start a conversation with me now.")
    
    chat = Chat(pairs, reflections)
    
    while True:
        user_input = input("> ")
        preprocessed_input = preprocess_input(user_input)

        if preprocessed_input in learned_data:
            print(learned_data[preprocessed_input])
        else:
            predictions = predict_class(preprocessed_input, model)

            if predictions:
                intent = predictions[0]["intent"]
                print(f"Intent: {intent}")

                # Generate a response based on the predicted intent
                response_generated = False

                for pattern, responses in pairs:
                    if re.match(pattern, intent):
                        print(random.choice(responses))
                        response_generated = True
                        break

                if not response_generated:
                    print("I'm still learning. What do you mean by that?")
                    user_answer = input("Your answer: ")
                    learned_data[preprocessed_input] = user_answer
            else:
                print("I'm still learning. What do you mean by that?")
                user_answer = input("Your answer: ")
                learned_data[preprocessed_input] = user_answer