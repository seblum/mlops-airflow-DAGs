FROM python:3.8

RUN pip install -f ./requirements.txt
RUN pip install mlflow

#RUN mlflow models serve --model-uri $MODEL_URI -h 0.0.0.0 -p $SERVING_PORT --no-conda
COPY scripts/serving_entrypoint.sh /serving_entrypoint.sh

ENTRYPOINT [ "/bin/bash", "/serving_entrypoint.sh" ]

# ENTRYPOINT [ "mlflow" "models" "serve" "--model-uri" "$MODEL_URI" "-h" "0.0.0.0" "-p" "$SERVING_PORT" "--no-conda" ]
