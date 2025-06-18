# Agent authentication code example in Databricks

Databricks has enabled behalf-of-user authentication in beta, finally! This means agents can now access Databricks resources using the identity of the Databricks end user who invoked the agent. The solution includes an example agent solution with the deployment process.

The article on the solution can be found here: 
* [Learn how to properly handle agent authentication in Databricks](https://www.ikidata.fi/post/learn-how-to-properly-handle-agent-authorization-in-databricks)

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites and Installation](#prerequisites-and-installation)
- [Usage](#usage)
- [Known limitations](#Limitations)

## Introduction

The solution contains an agent.py file, which includes the agent code. In the src folder, there are helper functions that the agent uses, as well as a config.py file. The deployment process can be found in the driver notebook, using the same logic that is used in Databricks documentation examples. Once again, easy-to-use demo agent code, can be used only for test purposes.

## Prerequisites and Installation 

Remember to add Genie space ID, Genie tool description and llm model endpoint name in config.py file. If you're wondering why it's not using YAML, there are some hiccups with path management during deployment, and this was my preferable solution to avoid extra manual work or latency due to multipath checking. In agent.py file you can change MLflow tracing destination and experiment name. Remember to add correct catalog and schema name in the cell 6 in driver notebook.

## Usage

![usage](https://static.wixstatic.com/media/729282_07d53dd477f84405b0487b8e926f6a0a~mv2.gif)

Run the driver notebook after setting up the config.py file and then the agent can be used normally via model serving endpoint. 

## Limitations
- Keep in mind that this is for demo purposes only, to demonstrate how easy it is to use behalf-of-user authentication with agents in Databricks.
- All validation has been done in Azure Databricks.

## Official documentation
https://docs.databricks.com/aws/en/generative-ai/agent-framework/author-agent?language=Genie+Spaces#on-behalf-of-user-authentication

## More information
To stay up-to-date with the latest developments: 
1) Follow Ikidata on LinkedIn: https://www.linkedin.com/company/ikidata/ 
2) Explore more on our website: https://www.ikidata.fi

![logo](https://github.com/ikidata/ikidata_public_pictures/blob/main/logos/Ikidata_aurora_small.png?raw=true)