import os
import requests
from typing import List

class TicketAPI:
    def __init__(self):
        self.env = os.environ.get("CODING_CHALLENGE_ENV", "host")
        self.host = "http://host.docker.internal:8080" if self.env == "docker" else "http://localhost:8080"
        self.token = os.environ["ZAMMAD_TOKEN"]

    def list_tickets(self):
        response = requests.get(
            f"{self.host}/api/v1/tickets",
            headers={"Authorization": f"Token token={self.token}"}
        )
        return response.json()

    def show_ticket(self, ticket_id):
        response = requests.get(
            f"{self.host}/api/v1/tickets/{ticket_id}",
            headers={"Authorization": f"Token token={self.token}"}
        )
        return response.json()

    def create_ticket(self, title):
        response = requests.post(
            f"{self.host}/api/v1/tickets",
            data={
                "title": title,
                "group_id": 1,
                "customer": "nicole.braun@zammad.org"
            },
            headers={"Authorization": f"Token token={self.token}"}
        )
        return response.json()

    def update_ticket(self, ticket_id, priority_id):
        response = requests.put(
            f"{self.host}/api/v1/tickets/{ticket_id}",
            data={
                "priority_id": priority_id,
            },
            headers={"Authorization": f"Token token={self.token}"}
        )
        return response.json()

    def create_ticket_article(self, ticket_id, subject, body_text):
        response = requests.post(
            f"{self.host}/api/v1/ticket_articles",
            data={
                "ticket_id": ticket_id,
                "subject": subject,
                "body": body_text,
                "content_type": "text/html",
                "type": "email",
                "internal": False,
                "sender": "Customer"
            },
            headers={"Authorization": f"Token token={self.token}"}
        )
        return response.json()

    def delete_ticket(self, ticket_id):
        response = requests.delete(
            f"{self.host}/api/v1/tickets/{ticket_id}",
            headers={"Authorization": f"Token token={self.token}"}
        )
        return response.status_code

    def create_ticket_with_article(self, title, subject, body_text):
        response = self.create_ticket(title)
        print(f"Response in api.create_ticket_with_article:\n{response}")
        ticket_id = response["id"]
        reponse_article = self.create_ticket_article(ticket_id, subject, body_text)
        return response, reponse_article

    def get_articles_by_ticket_id(self, ticket_id):
        response = requests.get(
            f"{self.host}/api/v1/ticket_articles/by_ticket/{ticket_id}",
            headers={"Authorization": f"Token token={self.token}"}
        )
        return response.json()


class TicketApiExtended(TicketAPI):
    def update_ticket_batch(self, ticket_ids: List[int], priorities: List[int]):
        assert len(ticket_ids) == len(
            priorities), f"Please provide same number of ticket_ids and priority_ids (ticket_ids: {len(ticket_ids)}, priority_ids: {len(priorities)})"
        for ticket_id, priority in zip(ticket_ids, priorities):
            print(f"Updating ticket {ticket_id} to priority {priority}")
            self.update_ticket(ticket_id, priority)


if __name__ == '__main__':
    api = TicketApiExtended(system="docker")
    print(api.list_tickets())