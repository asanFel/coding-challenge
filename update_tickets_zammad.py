from ticket_api import TicketApiExtended
from typing import Dict, List
from model import Classifier
from helpers import download_dataset


class Converter:
    @staticmethod
    def label_to_priority(label: int) -> int:
        mapping = {
            2: 1,
            1: 2,
            0: 3
        }
        return mapping[label]

    @staticmethod
    def priority_to_human_readable(priority: int) -> str:
        mapping = {
            1: "positive",
            2: "medium",
            3: "negative",
        }
        return mapping[priority]

    @staticmethod
    def label_to_human_readable(label: int) -> str:
        mapping = {
            2: "positive",
            1: "medium",
            0: "negative",
        }
        return mapping[label]


def get_complaints(ticket_api: TicketApiExtended) -> Dict[int, str]:
    tickets = get_all_tickets(ticket_api)
    complaints = {}
    for ticket in tickets:
        id = ticket['id']
        ticket_information = ticket_api.get_articles_by_ticket_id(id)
        complaint = ticket_information[0]['body']
        complaints[id] = complaint
    return complaints


def make_predictions(model: Classifier, complaints: List[str]) -> List[int]:
    predictions = model.predict_sentiment(complaints)
    return predictions


def update_tickets(ticket_api: TicketApiExtended, complaints: Dict[int, str], predictions: List[int], verbose: bool = False):
    assert len(predictions) == len(complaints), f"Please provide same number of predictions and complaints (issues: {len(predictions), complaints: {len(complaints)}})"
    for ticket_id, prediction in zip(list(complaints.keys()), predictions):
        priority = Converter.label_to_priority(prediction)
        print(f"Updating ticket {ticket_id} to priority {priority}")
        ticket_api.update_ticket(ticket_id, priority)


def file_tickets(issues: List[str], complaints: List[str], ticket_api: TicketApiExtended, verbose: bool = False):
    assert len(issues) == len(complaints), f"Please provide same number of issues and complaints (issues: {len(issues), complaints: {len(complaints)}})"
    for count, (issue, complaint) in enumerate(zip(issues, complaints)):
        title = f"Complaint_{count+1}"
        if verbose:
            print(f"Filing complaint {title}")
        _, _ = ticket_api.create_ticket_with_article(title, issue, complaint)


def get_all_tickets(ticket_api: TicketApiExtended):
    tickets = ticket_api.list_tickets()
    return tickets


def load_complaints(num_complaints: int = 50) -> (List, List):
    df = download_dataset(num_complaints)
    complaints = list(df["Consumer Complaint"].values)
    issues = list(df["Issue"].values)
    return issues, complaints
