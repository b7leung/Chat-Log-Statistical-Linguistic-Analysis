from posixpath import join
import ipywidgets as widgets
from ipywidgets import VBox, HBox
import os

class SentiAnalysisWidget():
    '''A sentiment analysis widget which classifies each user's message into one of five emotions.
    '''    

    def __init__(self, sentiments=['joy', 'sadness', 'fear', 'anger', 'neutral'], pic_size=420):
        '''Initalizes the sentiment analysis widget.

        :param sentiments: list of sentiments, defaults to ['joy', 'sadness', 'fear', 'anger', 'neutral']
        :type sentiments: list, optional
        :param pic_size: The size of the picture, defaults to 420
        :type pic_size: int, optional
        '''        
        self.pic_size = pic_size
        self.sentiments = sentiments
        self.num_senti = len(self.sentiments)
        self.cached_path = "./nlp_suite/sentiment_analysis/cached_user"
        self.users = os.listdir(self.cached_path)

        self.user_emo_html = """
            <font size="5">
            <p><b>Username:</b> {}</p>
            <p><b>General emotion:</b> {}</p>
            </font>
        """
        self.user_emo = widgets.HTML(self.user_emo_html.format("Unknown", "Unknown"))

        self.empty_graph = "./Graphics/loading2.png"

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

        self.top5_title = widgets.HTML("""<font size="5"><p><b>Show Top 5 sentences of:</b></p></font>""")

        self.top5_sent = [""] * 5
        self.top5_widget_HTML = '<font size="4"><b>{}</b></font><br><font size="3"><ul>{}</ul></font><br>'
        self.top5_widget = widgets.HTML("<p>No sentence found</p>")

        self.senti_boxes = [
            widgets.Checkbox(
                description=sent.capitalize(),
                disabled=True,
                indent=True,
                value=False
            ) for sent in self.sentiments
        ]

        for i, box in enumerate(self.senti_boxes):
            box.observe(self.update_selection)

        self.widget = VBox([
            self.user_emo,
            widgets.HTML('<font size="5"> <br> <b> Emotion distribution charts (Pie, Radar) </b> <br> <br> </font>'),
            HBox([
                self.emotion_distribution_pi,
                self.emotion_distribution_radar
            ]),
            widgets.HTML('<br>'),
            widgets.HTML('<br>'),
            self.top5_title,
            HBox(self.senti_boxes),
            self.top5_widget
        ])


    def get_widget(self):
        """Returns the underlying ipywidget

        :return: the sentiment analysis ipywidget
        :rtype: ipywidgets.HTML 
        """           
        return self.widget


    def update_selection(self, button):
        '''Updates the sentiment selection.

        :param button: A button instance object.
        :type button: ipywidgets.Button
        '''        
        top5 = ""
        for i, box in enumerate(self.senti_boxes):
            if box.value:
                if box.description == self.gen_emo.capitalize():
                    top5 = top5 + "<p>" + self.top5_widget_HTML.format(
                        self.sentiments[i].capitalize() + " <font color='blue'>(main sentiment)</font>",
                        "<li>" + "</li> <li>".join(self.top5_sent[i]) + "</li>"
                    ) + "</p>"
                else:
                    top5 = top5 + "<p>" + self.top5_widget_HTML.format(
                        self.sentiments[i].capitalize(),
                        "<li>" + "</li> <li>".join(self.top5_sent[i]) + "</li>"
                    ) + "</p>"
        self.top5_widget.value = top5


    def init_widget_data(self, user_info):
        '''Initalizes the chatbot with user-specific info.

        :param user_info: Pandas Dataframe
        :type user_info: pd.DataFrame
        '''        

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
            self.top5_sent = [top5_sent[i + 1 : i + self.num_senti + 1] for i in range(0, len(top5_sent), self.num_senti + 1)]

        self.user_emo.value = self.user_emo_html.format(user_info["user_name"], gen_emo.capitalize())
        self.gen_emo = gen_emo
        for i, box in enumerate(self.senti_boxes):
            box.disabled = False
            if box.description == gen_emo.capitalize():
                box.value = True
