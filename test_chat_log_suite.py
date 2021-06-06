import pickle
import os

import pytest

from nlp_suite import data_preprocessing
from basic_info_widget import BasicInfoWidget 
from stylized_chatbot_widget import StylizedChatbotWidget


# data_preprocessing tests

@pytest.fixture
def processed_discord_data():
    return data_preprocessing.process_discord_data(["test_files/[298954459172700181] [part 9].txt"], 3)

@pytest.mark.data_preprocessing
def test_processed_chat_data_msg_cnts(processed_discord_data):
    channel_messages, message_counts = processed_discord_data
    assert message_counts[:5] == [['angel', 4688], ['CreeatureCreator', 3464], ['4Chain', 3331], ['Larsundso', 3166], ['ErickAlvarez', 2601]]

@pytest.mark.data_preprocessing
def test_processed_chat_data_channel_msgs(processed_discord_data):
    channel_messages, message_counts = processed_discord_data
    assert channel_messages["angel"][:3] == ['lasruuu oop i cant type', 'i no sad  Heeyya Lars and Angel @Gali haiii gali', 'im fine thank you:3 how are you?']

# widgets tests

@pytest.fixture
def muffins_user_info():
    user_name = "muffins"
    user_messages = pickle.load(open("test_files/muffins_user_messages.p", "rb"))
    return {"user_name": user_name, "user_messages": user_messages}

# basic info widget tests

@pytest.mark.basic_info_widget
def test_basic_info_widget(muffins_user_info):
    widget = BasicInfoWidget()
    widget.init_widget_data(muffins_user_info)
    correct_text = ['','','','','','','','','<p><b>Username:</b>','muffins<p>\n','','','','','','','','<p><b>Number','of','Messages:</b>','15400<p>\n','','','','','','','','<p><b>Message','Samples:</b>','<br>woah:c','aussies','are','nice<br>i','have','aussie','friends',"they're",'very','nice','to','me<br>me,','buying','Pokemon','black','because','i','thought','it','had','zekrom<br>i','knoww','thats','even','dumber','omg<br>GHETSIS','EVIL','EYEBALL','MAN<br>byeee','Filipe!','hi!','do','i','call','you',"filipe?<br>someone's",'just','controlling','it','lmao<br>I','have','a','name<br>uwu!','good','morning','Zusty','Lemons<br>do','you..','do','you','like','Lemons?<p>\n\n','','','','','','','','']
    widget_text = widget.get_widget().value.split(' ')
    assert widget_text == correct_text


# chatbot tests

@pytest.mark.chatbot
def test_chatbot_widget(muffins_user_info):
    widget = StylizedChatbotWidget()
    assert widget.begin_button.disabled

    widget.init_widget_data(muffins_user_info)
    assert widget.user_name == "muffins"
    assert not widget.begin_button.disabled

    widget.reset()
    widget.restart(widget.restart_button)
    assert widget.text_box.disabled
    

@pytest.mark.chatbot_gpu
def test_chatbot_widget_gpu(muffins_user_info):
    widget = StylizedChatbotWidget()
    widget.init_widget_data(muffins_user_info)
    widget.setup_model_weights(None)
    assert widget.begin_button.description == "Chatbot Loaded!"
    assert widget.begin_button.button_style == "success"
    assert widget.restart_button.disabled == False

    chatbot_query_msg = "Hi, how are you?"
    widget.text_box.value = chatbot_query_msg
    widget.submit_to_chatbot(widget.text_box)
    assert chatbot_query_msg in widget.all_conversation_text
    assert len(widget.all_conversation_text.replace(chatbot_query_msg,"")) > 0

