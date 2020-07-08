FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple gym redis
WORKDIR /var
ADD actor.py /var
ENTRYPOINT ["python", "/var/actor.py"]
