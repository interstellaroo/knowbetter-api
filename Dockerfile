FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tell NLTK where the data will be
ENV NLTK_DATA=/usr/share/nltk_data

# Copy NLTK data from project directory (not absolute path)
COPY nltk_data/ /usr/share/nltk_data/

# Sanity check with proper syntax
RUN python -c "import nltk; nltk.data.find('tokenizers/punkt'); nltk.data.find('tokenizers/punkt_tab'); print('punkt + punkt_tab OK')"

COPY app/ ./app/
EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]