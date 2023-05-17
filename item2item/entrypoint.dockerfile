FROM python:3.9

WORKDIR /home

ADD . /home

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

CMD python3.9 entrypoint.py $P1 $P2 $P3 $P4 $P5 $P6 $P7