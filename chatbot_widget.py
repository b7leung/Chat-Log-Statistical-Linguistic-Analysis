
add_paths = ['nlp_suite/chatbot/style_transfer_paraphrase', 'nlp_suite/chatbot/style_transfer_paraphrase/style_paraphrase']
for add_path in add_paths: 
    if add_path not in sys.path: sys.path.append(add_path)

from nlp_suite.chatbot.style_transfer_chatbot import StyleTransferChatbot

blenderbot_chatbot_model = BlenderbotForConditionalGeneration.from_pretrained('facebook/blenderbot-400M-distill').to("cuda") # loading chatbot model

# NOTE: this takes about 1.5 mins to load. Not sure how to make faster
style_model_dir = "cached_user_data/{}/style_transfer_paraphrase_checkpoint".format(user_name)
chatbot = StyleTransferChatbot(style_model_dir, blenderbot_chatbot_model)
chatbot_tokenizer = chatbot.tokenizer

# messages seperator based on https://github.com/huggingface/transformers/issues/9365
seperator = "    "
all_conversation_text = ""
max_position_embeddings = 128

out = widgets.Output(layout={'border': '2px solid black', "height":"400px", "width":"900px","overflow":'scroll'})

@out.capture()
def submit_to_chatbot(next_user_input=""):
    global all_conversation_text
    
    # adding user input to conversation text
    print(colored('> User', 'blue', attrs=["bold"]) + ": {}".format(next_user_input))
    if all_conversation_text != "": 
        next_user_input = seperator + next_user_input
    all_conversation_text += next_user_input
    
    # truncate token length if needed
    curr_token_length = chatbot_tokenizer([all_conversation_text], return_tensors='pt')['input_ids'].shape[1]
    while curr_token_length >= max_position_embeddings:
        all_conversation_text = seperator.join(all_conversation_text.split(seperator)[1:])
        curr_token_length = chatbot_tokenizer([all_conversation_text], return_tensors='pt')['input_ids'].shape[1]
    
    # get response
    bot_stylized_response, bot_response = chatbot.get_response(all_conversation_text)
    print(colored('> Bot (original)', 'green', attrs=["bold"]) + ": {}".format(bot_response))
    print(colored('> Bot (stylized)', 'red', attrs=["bold"]) + ": {}".format(bot_stylized_response))
    print("")
    all_conversation_text += seperator + bot_stylized_response  

def restart(button_instance):
    global all_conversation_text
    all_conversation_text = ""
    out.clear_output()

button = widgets.Button(description='Restart Chat',disabled=False,button_style='')
button.on_click(restart)

text_box = widgets.Text(description="User Chat:")
text_input = interactive(submit_to_chatbot,{'manual': True, "manual_name":"Submit Message"}, next_user_input=text_box)
left = VBox([out,button])
display(HBox([text_input,left]))