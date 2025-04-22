from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="iGXfWgrgvjHd8mY7SUJF"
)

result = CLIENT.infer("garbage_5.jpeg", model_id="garbage_best/1")