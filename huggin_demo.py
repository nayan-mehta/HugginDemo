import docx
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return ' '.join(fullText)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# #1 Use for docx
# input_text = getText("/Users/nayanmehta/PycharmProjects/huginface/earnings_call_msft/human_generated/TranscriptFY18Q1.docx")
#
# input_text = "".join(input_text.split("\\n"))

# input_file = open("/Users/nayanmehta/PycharmProjects/huginface/earnings_call_msft/machine_formatted_text/Microsoft-Fiscal-Year-2018-First-Quarter-Earnings-Conference-Call.txt", "r")
# input_text = input_file.read()
#
# words = input_text.split()
# edits = []
# for word in words:
#     edits.append(word.replace("\n", ""))
#
# input_text = " ".join(edits)

# #2 Use relevant passage
input_text = "Our third quarter revenue was $26.8 billion, up 16 percent and 13 percent in constant currency, with better than expected performance across all segments.  Gross margin increased 16 and 13 percent in constant currency, operating income increased 23 percent and 20 in constant currency.  Earnings per share was 95 cents, increasing 36 percent and 31 percent in constant currency."

question = "What was revenue?"
# print(repr(input_text))

# TODO: Fix size the initializing the embedding layer with the right size i.e. Size of Vocabulary and not 512 (https://github.com/chenxijun1029/DeepFM_with_PyTorch/issues/1)
# https://github.com/huggingface/transformers/issues/1791

# input_text = tokenizer.encode(input_text, add_special_tokens=True, max_length=10)
# print(input_text[:512])
encoding = tokenizer.encode_plus(question, input_text[:512])
input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
print(answer)

# input_text = "The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano."
# question, text = "Who stars in The Matrix?", input_text

# assert answer == "Bla blee bloo"
