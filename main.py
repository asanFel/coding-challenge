from ticket_api import TicketApiExtended
from model import Classifier
from update_tickets_zammad import get_complaints, make_predictions, update_tickets, load_complaints, file_tickets


def main():
    ticket_api = TicketApiExtended()
    model = Classifier()

    # READ PARQUET FILE
    print("Load complaints data from file ...")
    issues, complaints = load_complaints()

    # CREATE TICKETS
    print("Creating tickets ...")
    file_tickets(issues, complaints, ticket_api)

    # GET ALL TICKETS
    print("Receiving tickets ...")
    complaints = get_complaints(ticket_api)
    print(complaints)

    # GET ALL COMPLAINTS AND MAKE PREDICTIONS
    print("Making predictions ...")
    predictions = make_predictions(model, list(complaints.values()))

    # UPDATE TICKETS
    print("Updating ticket priorities ...")
    update_tickets(ticket_api, complaints, predictions)


if __name__ == '__main__':
    main()
