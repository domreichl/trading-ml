import numpy as np

from utils.data_processing import stack_array_from_dict, get_final_predictions_from_dict


def test_stack_array_from_dict():
    dictionary = {"A": [0, 1, 2], "B": [3, 4, 5]}
    array = stack_array_from_dict(dictionary, 0)
    assert array.shape == (2, 3)
    array = stack_array_from_dict(dictionary, 1)
    assert array.shape == (3, 2)


def test_get_final_predictions_from_dict():
    dictionary = {"A": [10, 11, 12], "B": [13, 14, 15]}
    array = get_final_predictions_from_dict(dictionary)
    assert np.array_equal(array, [12, 15]) == True
