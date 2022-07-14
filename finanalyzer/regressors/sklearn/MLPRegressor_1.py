import sklearn

print(sklearn.__version__)


class MLRegressor:
    """Create a Machine Learning model for data regression"""

    def __init__(self, data):
        self.data = data
        self.model = self.create_model()

    def __str__(self):
        return self.represent()

    def __repr__(self):
        return self.represent()

    def represent(self):
        return self.model.

    def create_model(self):
        pass

    def train_model(self):
        pass


if __name__ == '__main__':
    data = get_data(filename, )
