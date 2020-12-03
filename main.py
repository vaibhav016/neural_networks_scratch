from model.model import Model


def load_data():
    pass


def create_model(input_shape):
    model = Model()
    # 2 layers of 3 nodes and 1 output layer
    model.add(number_of_neurons=3, input_shape=input_shape)
    model.add(number_of_neurons=3)
    model.add(number_of_neurons=1)


def train_model(model, data):
    pass


def run_neural_network(data):
    model = create_model()
    train_model(model, data)


def obtain_input_shape(data):
    pass


def main():
    data = load_data()
    input_shape = obtain_input_shape(data)
    run_neural_network(data)


if __name__ == "__main__":
    create_model(4)
