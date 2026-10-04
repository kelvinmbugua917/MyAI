import numpy as np

training_data = {
    "greeting": [
        "hello",
        "hi",
        "hey",
        "good morning",
        "how are you"
    ],

    "farewell": [
        "bye",
        "goodbye",
        "see you",
        "talk later"
    ],

    "identity": [
        "who are you",
        "what are you",
        "what is your name"
    ],

    "thanks": [
        "thank you",
        "thanks",
        "much appreciated"
    ]
}

responses = {
    "greeting": "Hello! I'm MyAI.",
    "farewell": "Goodbye! See you again.",
    "identity": "I'm MyAI, a small language model we're building.",
    "thanks": "You're welcome!"
}

# Convert intent names into numerical labels
intents = list(training_data.keys())

intent_to_label = {
    intent: index
    for index, intent in enumerate(intents)
}

print("Intents:", intents)
print("Labels:", intent_to_label)