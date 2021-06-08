import pickle
import os

import pytest
import numpy as np
import pandas as pd
import ipywidgets as widgets
from plotly.graph_objs import FigureWidget

from nlp_suite import data_preprocessing
import nlp_suite.clustering.preprocessing as cluster_preprocessing
from basic_info_widget import BasicInfoWidget 
from stylized_chatbot_widget import StylizedChatbotWidget
from cluster_widget import ClusterWidget
from senti_analysis_widget import SentiAnalysisWidget
from nlp_suite.clustering import utils


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


# cluster widget tests

@pytest.mark.cluster_widget
def test_cluster_widget(muffins_user_info):
    widget = ClusterWidget()
    widget.data_been_processed(muffins_user_info)
    correct_cluster = 4
    cluster = None
    for w in widget.get_widget().children:
        if isinstance(w, widgets.Dropdown): 
            cluster = w.value
    assert cluster == correct_cluster

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

# fixtures required to test  clustering utils

@pytest.fixture
def cluster_data():
    return pickle.load(open('nlp_suite/clustering/cluster_data.pkl', 'rb'))

@pytest.fixture
def encoder():
    return pickle.load(open('nlp_suite/clustering/encoder.pkl', 'rb'))

# cluster uitls tests

@pytest.mark.cluster_utils
def test_clustering_plot(cluster_data):
    pca_data, labels = cluster_data['pca'], cluster_data['labels']
    max_points = 20000
    point_size = 3
    plot = plot = utils.plot_3d_clusters(pca_data, labels, max_points=max_points, point_size=point_size)
    assert isinstance(plot, FigureWidget)
    assert len(plot.data) == 1
    assert len(plot.data[0].x) == len(plot.data[0].y) == len(plot.data[0].z) == max_points
    assert plot.data[0].marker.size == point_size


@pytest.mark.cluster_utils
def test_clustering_classifier(cluster_data, encoder):
    clusters = cluster_data['clusters']
    text_to_classify =  ['I really like video games, I am addicted to league of legends',
                         'mexican food is awesome.. burritos are the best']
    # true_labels = np.array([1,3])
    labels = utils.classify_text(encoder, clusters, text_to_classify)

    assert len(labels) == len(text_to_classify)
    assert all(labels < clusters.n_clusters)

@pytest.mark.cluster_utils
def test_cluster_preprocessing():
    datapath = './test_files'
    chats = cluster_preprocessing.get_chats(datapath, min_chats=10)
    assert isinstance(chats, pd.DataFrame)
    assert len(chats) == 480
    assert len(chats.columns) == 2


# sentiment analysis widget tests
@pytest.mark.sentiment_widget
def test_sentiment_widget(muffins_user_info):
    sentiments=['joy', 'sadness', 'fear', 'anger', 'neutral']
    pic_size=500
    widget = SentiAnalysisWidget(sentiments, pic_size)
    widget.init_widget_data(muffins_user_info)
    assert widget.num_senti == 5
    assert widget.user_emo_html == '\n            <font size="5">\n            <p><b>Username:</b> {}</p>\n            <p><b>General emotion:</b> {}</p>\n            </font>\n        '
    assert widget.pic_size == pic_size
    
    widget = widget.get_widget()
    assert len(widget.children) == 8

    


