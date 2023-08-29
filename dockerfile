FROM python:3.8-slim-buster

WORKDIR /app

COPY data.pickle /app/
COPY model.pt /app/
COPY author_to_index.pickle /app/
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
