class EarlyStopping:
    """
    Early stopping utility.

    Stops training when validation loss
    has not improved for a specified
    number of epochs.
    """

    def __init__(
        self,
        patience=5,
        min_delta=0.0,
    ):

        self.patience = patience

        self.min_delta = min_delta

        self.best_loss = float("inf")

        self.counter = 0

        self.should_stop = False

    def step(
        self,
        validation_loss,
    ):

        ####################################################
        # Improvement
        ####################################################

        if validation_loss < (

            self.best_loss - self.min_delta

        ):

            self.best_loss = validation_loss

            self.counter = 0

            return True

        ####################################################
        # No Improvement
        ####################################################

        self.counter += 1

        if self.counter >= self.patience:

            self.should_stop = True

        return False

    def reset(self):

        self.best_loss = float("inf")

        self.counter = 0

        self.should_stop = False