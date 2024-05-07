FROM python:3.8
ENV auth_url=https://auth.predictivehealthsolutions.co
ENV auth_api_key=b62cc0767b4c82a66ed82ad941b14f19499b84384d14d16952420c70f35ea597b0eef010882f4a657a73090b563a4bcd
EXPOSE 8501
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /app
COPY . ./
ENTRYPOINT ["streamlit", "run", "app.py",  "--server.address=0.0.0.0", "--server.enableCORS=false"]
