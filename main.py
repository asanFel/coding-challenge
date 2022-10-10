from ticket_api import TicketApiExtended
from model import Classifier
from update_tickets_zammad import get_complaints, make_predictions, update_tickets, load_complaints, file_tickets
import argparse


parser = argparse.ArgumentParser(description='coding challenge argparser.')
parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()


def main(args):
    ticket_api = TicketApiExtended()
    model = Classifier()

    # READ PARQUET FILE
    print("Load complaints data from file ...")
    issues, complaints = load_complaints()

    # CREATE TICKETS
    print("Creating tickets ...")
    file_tickets(issues, complaints, ticket_api, args.verbose)

    # GET ALL TICKETS
    print("Receiving tickets ...")
    complaints = get_complaints(ticket_api)
    print(complaints)

    # GET ALL COMPLAINTS AND MAKE PREDICTIONS
    print("Making predictions ...")
    predictions = make_predictions(model, list(complaints.values()))

    # UPDATE TICKETS
    print("Updating ticket priorities ...")
    update_tickets(ticket_api, complaints, predictions, args.verbose)


if __name__ == '__main__':
    main(args)