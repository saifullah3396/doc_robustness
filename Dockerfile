# Use nvidia/cuda image
FROM huggingface/transformers-pytorch-gpu:4.9.0

# install pip requirements
COPY ./requirements.txt /home/saifullah/requirements.txt
WORKDIR /home/saifullah
RUN pip3 install -r requirements.txt
