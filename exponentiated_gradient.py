import numpy as np

class ExponentialGradient:
    def test_case_1(self):
        data_set_1 = np.array([
            [0.343, 0.535, 0.412, -1.634, 0.237],
            [0.246, 0.432, -0.675, 1.293, 0.147]
        ])
        self.weighted_vote_method(data_set_1)

    def weighted_vote_method(self, data_set, learning_rate=0.3):
        if len(data_set) <= 0:
            raise ValueError("Data set length error.")

        weight_result = []

        # Initialize weight.
        current_weight = [1. / len(data_set[:,0]) for i in range(len(data_set[:,0]))]
        for i in range(len(data_set[0])):
            print("Current weight=\t\t" + str(current_weight))
            current_weight = self.exponentiated_gradient(data_set[:,i], current_weight, learning_rate)
            weight_result.append(current_weight)
            print("===================")

    def exponentiated_gradient(self, data_set, previous_weight, learning_rate):
        if len(data_set) <= 0:
            raise ValueError("Data set length error.")
        if len(data_set) != len(previous_weight):
            raise ValueError("Argument length not equal.")

        print("Data set in=\t\t" + str(data_set))

        result = []
        all_weighted_value = np.sum([previous_weight[i] * data_set[i] for i in range(len(data_set))])
        # weighted_value = [previous_weight[i] * data_set[i] for i in range(len(data_set))]
        # print weighted_value
        # all_weighted_value = 1.
        # for i in weighted_value:
        #     all_weighted_value *= i
        numerator = np.sum([previous_weight[i] * np.exp((learning_rate * data_set[i]) / all_weighted_value) for i in range(len(data_set))])
        print("Numerator=\t\t\t" + str(numerator))

        for i in range(len(data_set)):
            fractions = previous_weight[i] * np.exp((learning_rate * data_set[i]) / all_weighted_value)
            # print("Fractions=\t\t" + str(fractions))
            result.append(fractions / numerator)
        print("Result=\t\t\t\t" + str(result))
        print("Check result=\t\t" + str(np.sum(result)))
        return result

a = ExponentialGradient()
a.test_case_1()