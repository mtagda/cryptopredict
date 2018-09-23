FROM ubuntu:latest
MAINTAINER Magdalena Trzeciak magda.trzeciak@gmail.com
RUN apt-get update -y
RUN pip install -r requirements.txt
RUN apt-get install -y python-pip python-dev build-essential
ENTRYPOINT ["python"]
# add our project
ADD prediction.py /
# expose the port for the API
EXPOSE 5000
# run the API
CMD [ "python", "/prediction.py" ]


