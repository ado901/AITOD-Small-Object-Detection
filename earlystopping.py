import copy
# this class is used to stop the training process if the validation loss does not improve for a certain number of epochs
# this is useful to prevent overfitting and to save time, also it keeps the best model according to validation loss
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('inf')
        self.best_params = None
        self.early_stop = False

    def __call__(self, validation_loss,model):
        score = validation_loss
        if self.best_score - score > self.min_delta:
            self.best_score = score
            self.best_params = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True
        return self.early_stop