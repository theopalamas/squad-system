FROM python:3.11 as base

WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install -y build-essential libpq-dev gcc

RUN pip3 install --no-cache-dir poetry==1.2.2

COPY . .

ENV PATH = "${PATH}:/root/.poetry/bin"

# generate wheel and install application + dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction