FROM python:3.9.12
RUN pip install --upgrade pip
RUN pip install poetry==1.2.1
COPY ["../data/model.joblib", "/app/data/"]
WORKDIR /app
COPY ["../Docker/poetry.lock", "../Docker/pyproject.toml", "./"]
COPY ["../src/forest_ml/app/flask_app.py", "../src/forest_ml/app/__init__.py", "./"]
RUN poetry config virtualenvs.create false \
  && poetry install --without dev --no-interaction --no-ansi
EXPOSE 9696 
ENTRYPOINT [ "waitress-serve", "--host", "0.0.0.0", "--port", "9696", "app.flask_app:app" ]