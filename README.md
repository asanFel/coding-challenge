# coding-challenge
Repo for the coding challenge for deep opinion.<br>
Assumption: You have zammad running locally within docker-compose and you have an api-token with all the required rights
to make API calls.<br>
<br>
To run the code please follow the following steps:
1. Create a .env file in the coding-challenge folder and save the zammad token in the variable: `ZAMMAD_TOKEN` and create a second env variable called `CODING_CHALLENGE_ENV` and set it to `docker`
2. build docker container with: `docker build -t <name>:<version>`
3. run docker container with: `docker run -v "$PWD":/app --env-file=.env -t -d <name>:<version>`
4. enter the running docker container with: `docker exec -it <container_name> bash`
5. in the container execute: `python main.py` 

