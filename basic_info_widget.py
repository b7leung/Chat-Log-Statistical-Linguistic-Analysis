import ipywidgets as widgets

class BasicInfoWidget():
    '''Widget used to display basic user information
    '''    

    def __init__(self):
        '''Constructor Method for the basic info widget.
        '''             
        self.html = """\
        <p><b>Username:</b> {}<p>
        <p><b>Number of Messages:</b> {}<p>
        <p><b>Message Samples:</b> <br>{}<p>

        """
        self.widget = widgets.HTML(self.html.format("", "", ""))


    def get_widget(self):
        '''Obtain underlying widget object

        :return: Widget containing basic user information.
        :rtype: ipywidgets.HTML 
        '''        
        return self.widget


    def init_widget_data(self, user_info):
        '''Initialize Widget

        :param user_info: Dictionary containing messages for each user.
        :type user_info: dict
        '''        
        self.widget.value = self.html.format(user_info["user_name"], len(user_info["user_messages"]), "<br>".join(user_info["user_messages"][:10]))

