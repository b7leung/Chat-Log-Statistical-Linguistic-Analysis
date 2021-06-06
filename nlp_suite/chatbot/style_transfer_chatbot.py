import pprint
import os
import sys
import re

import torch
import nltk
nltk.download('punkt')
import transformers
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig

from style_paraphrase.inference_utils import GPT2Generator


class StyleTransferChatbot():
    '''Class representing a BlenderBot ChatBot with unsupervised style transfer. 
    Note that this needs about 9 GB DRAM, 6 GB GPU RAM.
    '''    

    # blenderbot_chatbot_model is an optional param, to speed things up
    def __init__(self, style_model_dir, blenderbot_chatbot_model=None):
        '''Initalizes the chatbot, and style transfer neural network.

        :param style_model_dir: The path to the weights for the style transfer neural network.
        :type style_model_dir: str
        :param blenderbot_chatbot_model: An optional blenderbot model, preloaded, to speed things up if available, defaults to None
        :type blenderbot_chatbot_model: BlenderBot, optional
        '''        

        transformers.logging.set_verbosity(transformers.logging.CRITICAL)
        
        # setting up chatbot components
        if blenderbot_chatbot_model is None:
            self.chatbot_model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill').to("cuda")
        else: 
            self.chatbot_model = blenderbot_chatbot_model
        self.tokenizer = BlenderbotTokenizer.from_pretrained('facebook/blenderbot-400M-distill')

        # setting up style transfer components; the inverse paraphraser and its hyperparameters
        with torch.cuda.device(0):
            self.style_model = GPT2Generator(style_model_dir)
        self.top_p_style = 0.7


    def get_response(self, input_text):
        '''Provides the response of the chatbot given an input utterance.

        :param input_text: The input query text to the chatbot. Note that it needs to be less than 128 chars after tokenization.
        :type input_text: str
        :return: Returns the stylized response, and original response
        :rtype: str, str
        '''        

        # getting initial chatbot response
        with torch.no_grad():
            input_tokens = self.tokenizer([input_text], return_tensors='pt').to('cuda')
            reply_ids = self.chatbot_model.generate(**input_tokens)
            raw_response = self.tokenizer.batch_decode(reply_ids)

            # cleaning response
            response = raw_response[0]
            response = response.replace("<s> ", "")
            response = response.replace("<s>", "")
            response = response.replace("</s>", "")

            # stylizing with persona
            with torch.cuda.device(0):
                stylized_response = []
                for sentence in nltk.tokenize.sent_tokenize(response):
                    stylized_sentence = self.style_model.generate_batch([sentence], top_p=self.top_p_style)[0]
                    stylized_response += stylized_sentence
                stylized_response = " ".join(stylized_response)

        return stylized_response, response

