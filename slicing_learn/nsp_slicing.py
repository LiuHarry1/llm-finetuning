from transformers import BertTokenizer, BertForNextSentencePrediction
import torch
import nltk

# Download punkt tokenizer for sentence splitting
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

# Load BERT NSP model and tokenizer
tokenizer = BertTokenizer.from_pretrained('C:/apps/ml_model/bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('C:/apps/ml_model/bert-base-uncased')
model.eval()

def is_next_sentence(sent1, sent2, threshold=0.5):
    """Use BERT NSP to determine if sent2 follows sent1."""
    encoding = tokenizer.encode_plus(sent1, sent2, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)  # [is_next, is_not_next]
        return probs[0, 0].item() >= threshold

def split_by_nsp(text, threshold=0.5):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return [text]  # return whole text if only one sentence

    chunks = []
    buffer = [sentences[0]]

    for i in range(1, len(sentences)):
        prev = buffer[-1]
        curr = sentences[i]
        if is_next_sentence(prev, curr, threshold):
            buffer.append(curr)
        else:
            chunks.append(" ".join(buffer))
            buffer = [curr]
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks

# Example text (e.g., from a Wikipedia article or documentation)
example_text = """
Artificial intelligence is transforming industries. Machines are now capable of performing complex tasks.
However, there are concerns about job displacement. Many experts suggest reskilling as a solution.
The history of AI goes back to the 1950s. Early researchers were optimistic about rapid progress.
"""

# Run NSP-based splitting
chunks = split_by_nsp(example_text, threshold=0.6)

# Print result
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:\n{chunk}")
