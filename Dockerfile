FROM ubuntu:latest
MAINTAINER Magdalena Trzeciak magda.trzeciak@gmail.com
COPY . .
WORKDIR .
RUN apt-get update
RUN apt-get install -y software-properties-common vim
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv
# update pip
RUN python3.6 -m pip install pip --upgrade
RUN python3.6 -m pip install wheel
RUN pip install -r requirements.txt
# add our project
ADD prediction.py /
# expose the port for the API
EXPOSE 5000
# run the API
ENTRYPOINT python3 prediction.py