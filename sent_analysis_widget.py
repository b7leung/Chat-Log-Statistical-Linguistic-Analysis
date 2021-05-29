from posixpath import join
import ipywidgets as widgets
from ipywidgets import VBox
import os

class SentiAnalysisWidget():

    def __init__(self, num_senti=5, pic_size=240):
        self.cached_path = "./nlp_suite/sentiment_analysis/cached_user"
        self.users = os.listdir(self.cached_path)
        self.pic_size = pic_size
        self.num_senti = num_senti

        self.user_emo_html = """
            <p><b>Username:</b> {}<p>
            <p><b>General emotion:</b> {}<p>
        """
        self.user_emo = widgets.HTML(self.user_emo_html.format("", ""))

        self.empty_graph = "./Graphics/logo_inverted.png"

        with open(self.empty_graph, "rb") as f_0:
            self.emotion_distribution_pi = widgets.Image(
                value=f_0.read(),
                format=self.empty_graph[-3:],
                width=self.pic_size,
                height=self.pic_size
            )

        with open(self.empty_graph, "rb") as f_0:
            self.emotion_distribution_radar = widgets.Image(
                value=f_0.read(),
                format=self.empty_graph[-3:],
                width=self.pic_size,
                height=self.pic_size
            )

        self.sentences_html = """
            <p><b>Top 5 Sentences of each emotion:</b>{}<p>
        """
        self.top5_sentences = widgets.HTML(self.sentences_html.format(""))
        self.widget = VBox([
            self.user_emo,
            widgets.HTML('<br> <b> Emotion distribution pie chart </b> <br>'),
            self.emotion_distribution_pi,
            widgets.HTML('<br> <b> Emotion distribution radar chart </b> <br>'),
            self.emotion_distribution_radar,
            widgets.HTML('<br>'),
            self.top5_sentences
        ])

    def get_widget(self):
        return self.widget

    def init_widget_data(self, user_info):

        assert user_info["user_name"] in self.users, "unknown user"
        user_path = os.path.join(self.cached_path, user_info["user_name"])

        with open(os.path.join(user_path, "label_result.txt"), "r") as f_eg:
            gen_emo = f_eg.readline().strip()

        with open(os.path.join(user_path, "pie_result.png"), "rb") as f_ed:
            self.emotion_distribution_pi.value = f_ed.read()

        with open(os.path.join(user_path, "radar_result.png"), "rb") as f_er:
            self.emotion_distribution_radar.value = f_er.read()

        with open(os.path.join(user_path, "top5_result.txt"), "r") as f_t5s:
            top5_sent = [line.strip() for line in f_t5s.readlines()]

        top5_sent = [("<br><br>" + "<br>".join(top5_sent[i * 6 : i * 6 + 5])) for i in range(self.num_senti)]
        top5_sent = "".join(top5_sent)
        self.user_emo.value = self.user_emo_html.format(user_info["user_name"], gen_emo)
        self.top5_sentences.value = self.sentences_html.format(top5_sent)


