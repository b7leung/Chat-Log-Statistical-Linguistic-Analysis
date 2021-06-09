# Introduction

With the advent of deep learning, new significant advances are being made at a rapid pace in the natural language processing (NLP) literature. However, for smaller companies, it can be difficult to integrate these new technologies without spending substantial resources on research and development. In our project we developed a Chat Log Language Analysis Suite to enable companies to readily understand the linguistic patterns of their user base. Many applications are integrated with social chatting systems, including video games, dating apps, and social media. With our product, even verbose chatters with thousands of logged chat messages can be summarized, evaluated, and compared with other users at a glance. Then, downstream use cases include flagging/suspending/banning toxic users, recommending advertisements or posts to users, and even using their “virtual” chatbot counterpart to predict their behavior to new inputs. 

Our presentation for this project can be found at this link: \
https://docs.google.com/presentation/d/1535V6VRLe-EIapHYMRdnkyu12eVziLMKEp6QLmh0_8M/edit?usp=sharing


# Documentation & Testing Framework

Our documentation is located under "Docs/_build/html". Note that contents under nlp_suite/chatbot/style_transfer_paraphrase are neither tested nor documented, as we did not develop that code. For more information, please visit [that project](https://github.com/martiansideofthemoon/style-transfer-paraphrase).

All tests can be run by: \
`pytest test_chat_log_suite.py`

If you don't have a GPU, you exclude the tests which require one: \
`pytest -m "not chatbot_gpu" test_chat_log_suite.py`

When running [coverage](https://github.com/nedbat/coveragepy) with all tests, using `coverage run --source=. -m pytest test_chat_log_suite.py` we currently achieve a **24% code coverage**: \
![image](https://user-images.githubusercontent.com/20059131/120935869-a5edb300-c6b9-11eb-8e1d-2256fc9ddb60.png)

One can also run coverage without the GPU tests with \
`coverage run --source=. -m pytest -m "not chatbot_gpu" test_chat_log_suite.py`, \
but this will result in a lower coverage.


# Example Data

## Example Inputs 
The following are some example inputs containing chat logs that can be uploaded to our NLP suite dashboard.

* User "muffins", discord server #298954459172700181 \
https://nlpsuite.s3-us-west-2.amazonaws.com/inputs/muffins.zip

* User "Saysora", discord server #691542050578890802 \
https://nlpsuite.s3-us-west-2.amazonaws.com/inputs/Saysora.zip

* User "circus", discord server #731254148678549595 \
https://nlpsuite.s3-us-west-2.amazonaws.com/inputs/circus.zip

## Cached Data
The following are pretrained style transfer transformer weights for the chatbot. They can be placed in the "cached_user_data" directory.

* User "muffins", discord server #298954459172700181 \
https://nlpsuite.s3-us-west-2.amazonaws.com/muffins.zip

* User "Saysora", discord server #691542050578890802 \
https://nlpsuite.s3-us-west-2.amazonaws.com/Saysora.zip

* User "circus", discord server #731254148678549595 \
https://nlpsuite.s3-us-west-2.amazonaws.com/circus.zip

# Deployment Setup

## With Chatbot
The NLP suite, with chatbot functionality, requires at least 6 GB of GPU memory. It can be set up is as follows.
* Launch an AWS EC2 instance (we used a p2.xlarge instance with the "Ubuntu Server 18.04 LTS (HVM), SSD Volume Type" AMI), and SSH into it.
* Install [NVIDIA drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html)
* Install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
* Clone this repo, and create a conda environment with the required dependencies: \
`conda env create --file deployment/nlp_suite_conda_env.yaml`
* Under the created conda environment, run a Jupyter Notebook, or alternatively, Voila: \
`jupyter notebook --no-browser --port=5666 --NotebookApp.token='' --NotebookApp.password='' --ip='0.0.0.0' --allow-root` \
or \
`voila dashboard.ipynb --no-browser --port 5666`
* Port forward from your EC2 instance to your local PC, in a linux or WSL terminal: \
`ssh -i path/to/private_key.pem -N -L 8081:localhost:5666 ubuntu@instance_ip_address`
* Now, you can view the NLP suite at http://localhost:8081/ on your local PC.

## Without Chatbot
The NLP suite can also be run on an instance without a GPU (say, t2.micro) but there will be no chatbot functionality. To do this, simply follow the instructions in the "with chatbot" case above, but omit installing NVIDIA drivers. 

