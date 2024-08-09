FROM python:3.12
WORKDIR /workspace/PaintGameToolKit
COPY . ./
RUN pip install -r requirements.txt
CMD ["python", "main.py"]