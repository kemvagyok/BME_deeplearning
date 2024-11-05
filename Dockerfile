FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

COPY . .

RUN apt-get update && apt install -y && apt-get install -y wget 
RUN python -m pip install -r requirements.txt 
RUN python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
RUN wget https://snap.stanford.edu/data/facebook.tar.gz 
RUN wget https://snap.stanford.edu/data/gplus.tar.gz
RUN wget https://snap.stanford.edu/data/twitter.tar.gz
RUN wget https://snap.stanford.edu/data/twitter_combined.txt.gz
RUN wget https://snap.stanford.edu/data/gplus_combined.txt.gz
RUN wget https://snap.stanford.edu/data/facebook_combined.txt.gz

SHELL ["/bin/bash", "-l", "-c"]

ENTRYPOINT python main.py