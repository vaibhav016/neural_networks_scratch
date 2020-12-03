from random import seed

from data.data import Data
from model.model import Model


def create_model(input_shape):
    model = Model()
    # 2 layers of 3 nodes and 1 output layer
    model.add(number_of_neurons=6, input_shape=input_shape)
    model.add(number_of_neurons=3)
    model.add(number_of_neurons=2)

    return model


def train_model(**kwargs):
    n_epochs = kwargs['n_epochs']
    data = kwargs['data']
    model = kwargs['model']
    n_outputs = kwargs['n_outputs']
    l_rate = kwargs['l_rate']

    for epoch in range(n_epochs):
        sum_error = 0
        for row in data:
            outputs = model.forward_propagation(row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1]-1)] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            model.backward_propagate_error(expected)
            model.update_weights(row, l_rate)
        print('>epoch=%d, l_rate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def run_neural_network(**kwargs):
    data = kwargs['data']
    sample_size = kwargs['sample_size']
    features = kwargs['features']
    n_epochs = kwargs['n_epoch']
    n_outputs = kwargs['n_outputs']
    l_rate = kwargs['l_rate']

    model = create_model(features)

    train_model(model=model, data=data, sample_size=sample_size, n_epochs=n_epochs, n_outputs=n_outputs, l_rate=l_rate)

    # for row in data:
    #     outputs = model.forward_propagation(row)
    #     prediction = outputs.index(max(outputs))
    #     print('Expected=%d, Got=%d' % (row[-1], prediction+1))


def main():
    data_obj = Data()
    data = data_obj.load_data()
    sample_size, features = data_obj.obtain_input_shape(data)
    # # seed(2)
    # data = [[2.7810836, 2.550537003, 0],
    #         [1.465489372, 2.362125076, 0],
    #         [3.396561688, 4.400293529, 0],
    #         [1.38807019, 1.850220317, 0],
    #         [3.06407232, 3.005305973, 0],
    #         [7.627531214, 2.759262235, 1],
    #         [5.332441248, 2.088626775, 1],
    #         [6.922596716, 1.77106367, 1],
    #         [8.675418651, -0.242068655, 1],
    #         [7.673756466, 3.508563011, 1]]
    # sample_size, features = len(data), 2
    n_outputs = 2
    l_rate = 0.5
    n_epoch = 20

    run_neural_network(data=data, sample_size=sample_size,
                       features=features, n_epoch=n_epoch,
                       l_rate=l_rate, n_outputs=n_outputs)




if __name__ == "__main__":
    main()
