from .testing_framework import TestingFramework
from .test_mytorch_softmax import test_softmax
from .test_mytorch_scaled_dot_product_attention import test_scaled_dot_product_attention
from .test_mytorch_multi_head_attention import test_multi_head_attention
from .test_mytorch_linear import test_linear
import json
import os


def test_lm_perplexity():
    print("Testing Language Modeling ...")
    # Check that model_arch.txt exists
    path_to_model_arch = "./model_arch.txt"
    assert os.path.exists(path_to_model_arch), "model_arch.txt file does not exist"
    print(f"Test passed: model_arch.txt")

    # Check that test_generated_results.json exists
    path_to_test_generated_results = "./test_generated_results.json"
    assert os.path.exists(path_to_test_generated_results), "test_generated_results.json file does not exist"
    print(f"Test passed: test_generated_results.json")

    # Check that the test_metrics.json file exists
    path_to_json = "./test_metrics.json"
    assert os.path.exists(path_to_json), "test_metrics.json file does not exist"

    # Check that the test_metrics.json file is a valid JSON file
    with open(path_to_json, 'r') as f:
        data = json.load(f)

    # Check that perplexity crosses threshold
    threshold = 3.5
    perplexity = data['perplexity_char']
    assert perplexity <= threshold, f"Character-level perplexity is greater than the threshold: {perplexity} > {threshold}"
    print(f"Test passed: Character-level perplexity ({perplexity}) <= Threshold ({threshold})")
    return data

if __name__ == "__main__":

    # Define the rubric for the tests
    rubric_dict = {
        "Linear": 5,
        "Softmax": 5,
        "ScaledDotProductAttention": 10,
        "MultiHeadAttention": 10,
        "LanguageModel": 75,
    }

    testing_framework = TestingFramework(
        test_categories={k:[] for k in rubric_dict.keys()}
    )

    # Register Linear Tests
    testing_framework.register_test_case("Linear", test_linear, "Linear Tests")

    # Register Softmax Tests
    testing_framework.register_test_case("Softmax", test_softmax, "Softmax Tests")

    # Register ScaledDotProductAttention Tests
    testing_framework.register_test_case("ScaledDotProductAttention", test_scaled_dot_product_attention, "ScaledDotProductAttention Tests")

    # Register MultiHeadAttention Tests
    testing_framework.register_test_case("MultiHeadAttention", test_multi_head_attention, "MultiHeadAttention Tests")

    # Register Language Modelling Tests
    testing_framework.register_test_case("LanguageModel", test_lm_perplexity, "LanguageModel Tests")

    # Run all tests
    testing_framework.run_tests()

    # Summarize results
    testing_framework.summarize_results()

    # Get auto results
    auto_results = testing_framework.get_autoresults(rubric_dict)
    print(json.dumps(auto_results))