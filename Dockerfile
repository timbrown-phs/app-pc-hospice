FROM python:3.8
ENV auth_url=https://33793539.propelauthtest.com
ENV auth_api_key=8dbd0a58e6369c541a2fc08ff973803e5c7c50555ff123bb767e7d9d7e7c227bf5376559d7687a62806c9846b6e90be8
EXPOSE 8501
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /app
COPY . ./
ENTRYPOINT ["streamlit", "run", "app.py",  "--server.address=0.0.0.0", "--server.enableCORS=false"]
