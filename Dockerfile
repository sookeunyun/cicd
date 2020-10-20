FROM python:3.7

RUN pip3 install matplotlib smart_open[s3] tensorflow==1.15.4 keras==2.0.8 flask
COPY app.py .

CMD ["python", "app.py"]

