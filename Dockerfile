FROM ubuntu:latest
MAINTAINER Magdalena Trzeciak magda.trzeciak@gmail.com
RUN apt-get update -y
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
# add our project
ADD . /
# expose the port for the API
EXPOSE 5000
# run the API
CMD [ "python", "/prediction.py" ]

