FROM borda/docker_python-opencv-ffmpeg:cpu-py3.7-cv4.4.0

RUN pip install numpy

ADD examples /examples
ADD main.py /main.py

CMD ["python", "/main.py"]