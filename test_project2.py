from project2 import cuisin_predict
import argparse

def test_cuisin_predict():

    # creating the N and ingredient values
    args = argparse.Namespace(N = "5", ingredient = ["white bread", "white onion", "grape tomatoes", "vegetable oil"])

    # Call the function with the test arguments
    output_dict = cuisin_predict(args)

    # Assert that the output dictionary has the expected keys and values
    assert set(output_dict.keys()) == {'cuisine', 'score', 'closest'}
    # breakpoint()
    assert isinstance(output_dict['cuisine'], str)
    assert isinstance(output_dict['score'], float)
    assert isinstance(output_dict['closest'], list)
    assert all(isinstance(d, dict) for d in output_dict['closest'])
    assert all(set(d.keys()) == {'id', 'score'} for d in output_dict['closest'])
    assert all(isinstance(d['id'], str) for d in output_dict['closest'])
    assert all(isinstance(d['score'], float) for d in output_dict['closest'])
