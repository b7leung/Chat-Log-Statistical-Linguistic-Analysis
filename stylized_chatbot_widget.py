import os
import sys

from ipywidgets import TwoByTwoLayout
from ipywidgets import Button, Layout
from termcolor import colored
from ipywidgets import interact, interactive, fixed, interact_manual,Button, HBox, VBox
import ipywidgets as widgets
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, BlenderbotConfig

add_paths = ['nlp_suite/chatbot/style_transfer_paraphrase', 'nlp_suite/chatbot/style_transfer_paraphrase/style_paraphrase']
for add_path in add_paths: 
    if add_path not in sys.path: sys.path.append(add_path)
from nlp_suite.chatbot.style_transfer_chatbot import StyleTransferChatbot


class StylizedChatbotWidget:
    """This provides and manages a ipywidget for a persona guided generative chatbot.
    """    

    def __init__(self):
        """Sets up the chatbot UI skeleton. 
        """        
        # messages seperator based on https://github.com/huggingface/transformers/issues/9365
        self.seperator = "    "
        self.all_conversation_text = ""
        self.max_position_embeddings = 128
        self.blenderbot_chatbot_model = None
        self.chatbot = None
        self.user_name = ""

        # setting up widget UI
        self.begin_button = widgets.Button(description='Load Chatbot', disabled=True, button_style='info')
        self.begin_button.on_click(self.setup_model_weights)
        self.text_box = widgets.Text(description="User Chat:", disabled=True)
        self.text_box.on_submit(self.submit_to_chatbot)
        self.out = widgets.Output(layout={'border': '2px solid black', "height":"450px", "width":"750px","overflow":'scroll'})
        self.restart_button = widgets.Button(description='Restart Chat', disabled=True, button_style='')
        self.restart_button.on_click(self.restart)
        self.chatbot_widget = HBox([VBox([self.begin_button, self.text_box]), VBox([self.out, self.restart_button])])
    

    def reset(self):
        """Resets the state of the chatbot. Meant to be called for each new user.
        """        
        self.text_box.value = ""
        self.all_conversation_text = ""
        self.out.clear_output()
        self.begin_button.description = 'Load Chatbot'
        self.begin_button.button_style = "info"
        self.text_box.disabled = True
        self.restart_button.disabled = True
        if self.chatbot is not None:
            del self.chatbot
    

    def get_widget(self):
        """Returns the underlying ipywidget

        :return: the chatbot ipywidget
        :rtype: ipywidgets.HTML 
        """        
        return self.chatbot_widget


    def init_widget_data(self, user_info):
        """Initalizes the chatbot with user-specific info.

        :param user_info: A dataframe of the user's information, with the username.
        :type user_info: Pandas Dataframe
        :return: the chatbot ipywidget
        :rtype: ipywidgets.HTML 
        """        
        self.reset()
        self.user_name = user_info["user_name"]
        self.begin_button.disabled = False
        return self.chatbot_widget


    def setup_model_weights(self, button_instance):
        """Sets up the chatbot neural network model's weights.

        :param button_instance: A button instance object
        :type button_instance: ipywidgets.Button
        """        
        self.begin_button.description = "Loading chatbot..."
        self.begin_button.button_style = "warning"
        self.begin_button.disabled = True
        if self.blenderbot_chatbot_model is None:
            self.blenderbot_chatbot_model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill').to("cuda") 
        style_model_dir = "cached_user_data/{}/style_transfer_paraphrase_checkpoint".format(self.user_name)
        self.chatbot = StyleTransferChatbot(style_model_dir, self.blenderbot_chatbot_model)
        self.chatbot_tokenizer = self.chatbot.tokenizer
        self.begin_button.description = "Chatbot Loaded!"
        self.begin_button.button_style = "success"
        self.restart_button.disabled = False
        self.text_box.disabled = False


    def submit_to_chatbot(self, text_instance):
        """Submits new text to the chatbot. That text, and the chatbot response, will be displayed on
        the output widget.

        :param text_instance: The text instance widget.
        :type text_instance: ipywidgets.Button
        """        
        next_user_input = text_instance.value
        self.text_box.value = ""
        
        # adding user input to conversation text
        self.out.append_stdout(colored('> User', 'blue', attrs=["bold"]) + ": {}\n".format(next_user_input))
        if self.all_conversation_text != "": 
            next_user_input = self.seperator + next_user_input
        self.all_conversation_text += next_user_input
        
        # truncate token length if needed
        curr_token_length = self.chatbot_tokenizer([self.all_conversation_text], return_tensors='pt')['input_ids'].shape[1]
        while curr_token_length >= self.max_position_embeddings:
            self.all_conversation_text = self.seperator.join(self.all_conversation_text.split(self.seperator)[1:])
            curr_token_length = self.chatbot_tokenizer([self.all_conversation_text], return_tensors='pt')['input_ids'].shape[1]
        
        # get response
        bot_stylized_response, bot_response = self.chatbot.get_response(self.all_conversation_text)
        self.out.append_stdout(colored('> Bot (original)', 'green', attrs=["bold"]) + ": {}\n".format(bot_response))
        self.out.append_stdout(colored('> Bot (stylized)', 'red', attrs=["bold"]) + ": {}\n".format(bot_stylized_response))
        self.out.append_stdout("\n")
        self.all_conversation_text += self.seperator + bot_stylized_response  
    

    def restart(self, button_instance):
        """Restarts the discussion history with the current chatbot.

        :param button_instance: A button instance object
        :type button_instance: ipywidgets.Button
        """        
        self.text_box.value = ""
        self.all_conversation_text = ""
        self.out.clear_output()
        
