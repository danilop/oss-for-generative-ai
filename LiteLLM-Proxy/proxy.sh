export AWS_REGION=us-east-1
#litellm --config ./config.yaml --port 8000 --host 0.0.0.0
litellm --model bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0 --port 8000 --host 0.0.0.0 # --drop_params --debug