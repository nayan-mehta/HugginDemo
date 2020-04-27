from transformers import pipeline
from transformers import XLNetTokenizer, XLNetForQuestionAnswering
import docx
import time


def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return ' '.join(fullText)

input_text = getText("/Users/nayanmehta/PycharmProjects/huginface/earnings_call_msft/human_generated/TranscriptFY18Q4.docx")
input_text = "".join(input_text.split("\\n"))

words = input_text.split()
edits = []
for word in words:
    edits.append(word.replace("\n", ""))

input_text = " ".join(edits)

# tokenizer =  XLNetTokenizer.from_pretrained('xlnet-base-cased')
# model = XLNetForQuestionAnswering.from_pretrained('xlnet-base-cased')

questions = [
"What was total revenue for the quarter?",
"What was the percent change in revenue for the quarter?",
"Was performance for the quarter better than expected?",
"What was earnings per share for the quarter?",
"What was the percent change in earnings per share for the quarter?",
"What are some business segments, products, or services that contributed to an increase in revenue?",
"Did foreign exchange, or FX, increase or decrease revenue?",
"What are some of the top opportunities for growth?",
"What ongoing or continuing investments have led to positive results?",
"Where will the firm increase investments to lead to future growth?",
"What investments increased operating expenses?"
]

qa = pipeline('question-answering', model="bert-large-cased-whole-word-masking-finetuned-squad", tokenizer= "bert-large-cased-whole-word-masking-finetuned-squad")
for question in questions:
    start_time = time.time()
    response = qa(context=input_text, question=question)
    stop = time.time()
    print("{0} {1} {2}".format(question , response['answer'],stop - start_time ))