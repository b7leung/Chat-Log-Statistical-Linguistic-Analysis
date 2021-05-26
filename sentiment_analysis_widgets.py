import ipywidgets as widgets
from ipywidgets import HBox, VBox
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import torch
from PIL import Image

class SentimentAnalysisWidget():
    def __init__(self):
        self.widget = widgets.Text(description="")
        self.tokenizer = None
        self.model = None
        self.user_name = ''
        self.class_names = ['joy', 'sadness', 'fear', 'anger', 'neutral']
        self.begin_button = widgets.Button(description='Load Model', disabled=True, button_style='info')
        self.begin_button.on_click(self.load_model)

        self.emotion_button = widgets.Button(description='Overall Emotion', disabled=True, button_style='info')
        self.emotion_button.on_click(self.show_emotion)

        self.pie_button = widgets.Button(description='Pie Chart', disabled=True, button_style='info')
        self.pie_button.on_click(self.show_pie)

        self.radar_button = widgets.Button(description='Radar Chart', disabled=True, button_style='info')
        self.radar_button.on_click(self.show_radar)

        self.top5_button = widgets.Button(description='Top Five Sentence of Emotion', disabled=True, button_style='info')
        self.top5_button.on_click(self.show_top5)

        self.clear_button = widgets.Button(description='Clear Output', disabled=True, button_style='')
        self.clear_button.on_click(self.restart)
        
        self.out = widgets.Output(layout={'border': '2px solid black', "height":"450px", "width":"750px","overflow":'scroll'})
        self.sentiment_widget = VBox([VBox([VBox([self.begin_button, HBox([HBox([HBox([self.emotion_button, self.pie_button]), self.radar_button]), self.top5_button])]), self.out]), self.clear_button])
    

    def get_widget(self):
        return self.sentiment_widget

    def init_widget_data(self, user_info):
        self.user_name = user_info["user_name"]
        self.begin_button.disabled = False
        self.emotion_button.disabled = False
        self.pie_button.disabled = False
        self.radar_button.disabled = False
        self.top5_button.disabled = False
        return self.sentiment_widget


    def load_model(self, button_instance):
        self.begin_button.description = "Loading model..."
        self.begin_button.button_style = "warning"
        self.begin_button.disabled = True
        if ((self.user_name != 'muffins') and (self.user_name != 'circus') and (self.user_name != 'Saysora')):
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(self.class_names), output_attentions=False,output_hidden_states=False)
            #model_dir = "./nlp_suite/sentiment_analysis/model/online.pt“
            self.model.load_state_dict(torch.load('./nlp_suite/sentiment_analysis/model/online.pt', map_location=torch.device('cpu')))
        self.begin_button.description = "Model Loaded!"
        self.begin_button.button_style = "success"
        #self.restart_button.disabled = False

    def show_emotion(self, button_instance):
        #self.begin_button.disabled = True
        with self.out:
            #label_path = "./nlp_suite/sentiment_analysis/cached_user/muffins/label_result.txt"
            label_path = "./nlp_suite/sentiment_analysis/cached_user/{}/label_result.txt".format(self.user_name)
            f = open(label_path)
            lines = f.readlines()
            for line in lines:
                print(line)
            print('!')
            f.close()
            
    def show_pie(self, button_instance):
        with self.out:
            pie_path = "./nlp_suite/sentiment_analysis/cached_user/{}/pie_result.png".format(self.user_name)
            image = Image.open(pie_path)
            image.show()
    
    def show_radar(self, button_instance):
        with self.out:
            radar_path = "./nlp_suite/sentiment_analysis/cached_user/{}/radar_result.png".format(self.user_name)
            image = Image.open(radar_path)
            image.show()

    def show_top5(self, button_instance):
        with self.out:
            top5_path = "./nlp_suite/sentiment_analysis/cached_user/{}/top5_result.txt".format(self.user_name)
            f = open(top5_path)
            lines = f.readlines()
            for line in lines:
                print(line)
            f.close()
    
    def restart(self, button_instance):
        self.out.clear_output()
