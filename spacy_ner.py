import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example
import random

# Load pre-trained English model
nlp = spacy.load("en_core_web_sm")

# Training data with new entity "TECH_COMPANY"
TRAIN_DATA = [
    ("OpenAI is developing new AI models.", {"entities": [(0, 6, "TECH_COMPANY")]}),
    ("Google and Microsoft are competing in AI.", {"entities": [(0, 6, "TECH_COMPANY"), (11, 20, "TECH_COMPANY")]}),
    ("Tesla is a leader in AI-powered robotics.", {"entities": [(0, 5, "TECH_COMPANY")]}),
    ("Apple is launching a new AI chip.", {"entities": [(0, 5, "TECH_COMPANY")]}),
    ("Meta is investing in the metaverse and AI.", {"entities": [(0, 4, "TECH_COMPANY")]}),
]

# Get the Named Entity Recognition (NER) pipeline
ner = nlp.get_pipe("ner")

# Add the new entity label
ner.add_label("TECH_COMPANY")

# Disable other pipeline components to only train NER
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

# Training the model
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.resume_training()
    
    for _ in range(20):  # Training for multiple epochs to improve learning
        random.shuffle(TRAIN_DATA)
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.3)  # Dropout helps prevent overfitting

# Save the updated model (optional)
nlp.to_disk("custom_ner_model")

# Test the trained model
test_text = "OpenAI and Tesla are investing in generative AI."
doc = nlp(test_text)

print("Recognized Entities:")
print([(ent.text, ent.label_) for ent in doc.ents])  # Should detect "OpenAI" and "Tesla" as TECH_COMPANY
