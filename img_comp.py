from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="0IwFXQMr4i2FQ4HpThQd"
)

result = CLIENT.infer(r"C:\Users\user\Downloads\fake nike shoe.jpg", model_id="counterfeit-nike-shoes-detection/2")
print(result)