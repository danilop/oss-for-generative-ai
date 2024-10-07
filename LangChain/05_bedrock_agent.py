import boto3
import json
import time
import uuid

from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain_aws.agents import BedrockAgentsRunnable


AWS_REGION = 'us-east-1'


@tool("AssetDetail::getAssetValue")
def get_asset_value(asset_holder_id: str) -> str:
    """
        Get the asset value for an owner id

        Args:
            asset_holder_id: The asset holder id

        Returns:
            The asset value for the given asset holder
    
    """
    return f"The total asset value for {asset_holder_id} is 100K"

@tool("AssetDetail::getMortgageRate")
def get_mortgage_rate(asset_holder_id: str, asset_value: str) -> str:
    """
        Get the mortgage rate based on asset value

        Args:
            asset_holder_id: The asset holder id
            asset_value: The value of the asset

        Returns:
            The interest rate for the asset holder and asset value
        
    """
    return (
        f"The mortgage rate for {asset_holder_id} "
        f"with asset value of {asset_value} is 8.87%"
    )

tools = [get_asset_value, get_mortgage_rate]

foundational_model = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

instructions="You are an agent who helps with getting the mortgage rate based on the current asset valuation"


def _create_agent_role(
        agent_region,
        foundational_model
) -> str:
    """
    Create agent resource role prior to creation of agent, at this point we do not have agentId, keep it as wildcard

    Args:
        agent_region: AWS region in which is the Agent if available
        foundational_model: The model used for inference in AWS BedrockAgents
    Returns:
       Agent execution role arn
    """
    try:
        account_id = boto3.client('sts').get_caller_identity().get('Account')
        assume_role_policy_document = json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "bedrock.amazonaws.com"
                    },
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:bedrock:{agent_region}:{account_id}:agent/*"
                        }
                    }
                }
            ]
        })
        managed_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AmazonBedrockAgentBedrockFoundationModelStatement",
                    "Effect": "Allow",
                    "Action": "bedrock:InvokeModel",
                    "Resource": [
                        f"arn:aws:bedrock:{agent_region}::foundation-model/{foundational_model}"
                    ]
                }
            ]
        }
        role_name = f'bedrock_agent_{uuid.uuid4()}'
        iam_client = boto3.client('iam')
        response = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=assume_role_policy_document,
            Description='Role for Bedrock Agent'
        )
        iam_client.put_role_policy(
            RoleName=role_name,
            PolicyName=f'AmazonBedrockAgentBedrockFoundationModelPolicy_{uuid.uuid4()}',
            PolicyDocument=json.dumps(managed_policy)
        )
        time.sleep(2)
        return response.get('Role', {}).get('Arn', '')

    except Exception as exception:
        raise exception

print("Creating agent role...")

agent_resource_role_arn = _create_agent_role(
            agent_region=AWS_REGION,
            foundational_model=foundational_model)

print(f"Agent resource role ARN: {agent_resource_role_arn}")

print("Creating agent...")

agent = BedrockAgentsRunnable.create_agent(
    agent_name="mortgage_interest_rate_agent",
    agent_resource_role_arn=agent_resource_role_arn,
    foundation_model=foundational_model,
    instruction="You are an agent who helps with getting the mortgage rate based on the current asset valuation",
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True) 

print("Invoking agent...")

output = agent_executor.invoke({"input": "what is my mortgage rate for id AVC-1234"})

print(output)


def _delete_agent_role(agent_resource_role_arn: str):
    """
    Delete agent resource role

    Args:
       agent_resource_role_arn: Associated Agent execution role arn
    """
    try:
        iam_client = boto3.client('iam')
        role_name = agent_resource_role_arn.split('/')[-1]
        inline_policies = iam_client.list_role_policies(RoleName=role_name)
        for inline_policy_name in inline_policies.get('PolicyNames', []):
            iam_client.delete_role_policy(
                RoleName=role_name,
                PolicyName=inline_policy_name
            )
        iam_client.delete_role(
            RoleName=role_name
        )
    except Exception as exception:
        raise exception


def _delete_agent(agent_id):
    bedrock_client = boto3.client('bedrock-agent')
    bedrock_client.delete_agent(agentId=agent_id, skipResourceInUseCheck=True)


_delete_agent(agent_id=agent.agent_id)
_delete_agent_role(agent_resource_role_arn=agent_resource_role_arn)
