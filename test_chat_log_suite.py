import pytest
import os
from nlp_suite import data_preprocessing


@pytest.fixture
def processed_discord_data():
    return data_preprocessing.process_discord_data(["test_files/[298954459172700181] [part 9].txt"], 3)

@pytest.mark.data_preprocessing
def test_process_chat_data(processed_discord_data):
    channel_messages, message_counts = processed_discord_data
    assert type(channel_messages) == dict
