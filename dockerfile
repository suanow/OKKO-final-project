FROM python:3.9

WORKDIR /home

ADD . /home

RUN pip install -r requirements.txt

EXPOSE 80

CMD python3.9 entrypoint.py $P1 $P2 $P3