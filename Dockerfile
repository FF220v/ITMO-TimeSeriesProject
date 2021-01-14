FROM python:3.8.7-buster
COPY ./src /src
ENV PYTHONPATH "/"
RUN pip3 install -r /src/requirements.txt
RUN ["chmod", "+x", "/src/run_worker.sh"]
EXPOSE 8050
COPY src/assets/* /src/assets/