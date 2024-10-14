FROM python

RUN apt-get update &&  \
	apt install -y \
	--mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

RUN wget https://snap.stanford.edu/data/facebook.tar.gz 
RUN wget https://snap.stanford.edu/data/gplus.tar.gz
RUN wget https://snap.stanford.edu/data/twitter.tar.gz

COPY . .	

SHELL ["/bin/bash", "-l", "-c"]

ENTRYPOINT python main.py