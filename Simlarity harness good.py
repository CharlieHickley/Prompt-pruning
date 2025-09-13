from levenshtein import levenshteinSimilarityCharacter
from levenshtein import levenshteinSimilarityWord
from similarity import get_similarity as cosineSimilarity
from ncd import ncd
import matplotlib.pyplot as plt
from yaml import safe_load_all, YAMLError
import openai
import pandas
from statistics import median

openai.api_key = ""

k = 10

threshold = None # currently not used


def read_prompts(path="C:/Users/clwhi/PycharmProjects/pythonProject/good prompts/prompt50.yaml"):
    """
    Reads the prompt file and returns the prompts in list

    Args:
        path (str, optional): The path to prompt file. Defaults to './data/prompt50.yaml'.

    Returns:
        list: list of prompts
    """

    try:
        with open(path, "r", encoding='utf-8') as file:

            try:
                data = list(safe_load_all(file))
                prompts = data[0]["prompts"]
                return prompts

            except YAMLError as exc:
                print(exc)
                exit(1)

    except FileNotFoundError:
        print("No data file found")
        exit(1)


def get_response(prompt, act, model="gpt-4o-mini"):
    output = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Act like you are" + act},
            {"role": "user", "content": prompt},
        ],
    )
    response = output.choices[0].message.content
    return response


def get_responses(prompt, act, model="gpt-4o-mini"):
    """
    Call openai api to get a list of random choices

    Args:
        prompt (str): The content of the prompt
        act (str): What the model should act as
        model (str, optional): The model from openAI API. Defaults to "gpt-4o-mini".

    Returns:
        list: list of responses
    """
    output = openai.ChatCompletion.create(
        model=model,
        n= k + 1,
        messages=[
            {"role": "system", "content": "Act like you are" + act},
            {"role": "user", "content": prompt},
        ],
    )
    responses = [choice.message.content for choice in output.choices]
    return responses


def generate_set_of_similarity(prompt, act, similarity_meassure, model="gpt-4o-mini"):
    """
    Generate a sample set of similarity under a meassure

    Args:
        prompt (str): The content of the prompt
        act (str): What the model should act as
        similarity_meassure (function): The similarity meassure function
        model (str, optional): The model from OpenAI API. Defaults to "gpt-4o-mini".

    Returns:
        list: 1D array of similarity score data
    """
    try:
        reponses = get_responses(prompt, act, model)
    except openai.error.OpenAIError as exc:
        print(exc)
        return None

    subset_1 = reponses[0]
    subset_2 = reponses[1:]

    set_of_similarity = [
        similarity_meassure(subset_1, subset_2i) for subset_2i in subset_2
    ]
    print(f"\n{act}: {str(set_of_similarity)}")
    return set_of_similarity


def plot(data):
    """plot the boxplot

    Args:
        data (ideally 2D array): The data to be plotted
    """
    plt.title("Sample sets")
    plt.xlabel("Similarity")
    plt.ylabel("Sample")
    plt.boxplot(data, vert=False)
    plt.show()


def any_result_matches(prompt, act, similarity_meassure, model="gpt-4o-mini"):
    """check if one of k responses is similar to the first response

    Args:
        prompt (str): The content of the prompt
        act (str): What the model should act as
        similarity_meassure (function): The similarity meassure function
        model (str, optional): The model from openAI API. Defaults to "gpt-4o-mini".

    Returns:
        Boolean: True if one of the responses is similar to the first response, False otherwise
        None: if an error occured
    """
    try:
        responses = get_responses(prompt, act, model)
    except openai.error.OpenAIError as exc:
        print(exc)
        return None

    response = responses[0]
    for i in range(1, len(responses) - 1):
        similarity = similarity_meassure(response, responses[i])

        if similarity >= threshold:
            print("correct +1")
            return True
        else:
            print("similarity: " + str(similarity))

    print("incorrect +1")
    return False


def main(similarity_meassure):

    prompts = read_prompts()
    sample_set = [
        generate_set_of_similarity(prompt["prompt"], prompt["act"], similarity_meassure)
        for prompt in prompts
    ]

    medianList = []
    for i in sample_set:
        medianList.append(median(i))
    data = {
        "median Stats": medianList
    }

    df = pandas.DataFrame(data)

    print(df.describe())
    plot(sample_set)


if __name__ == "__main__":
    main(cosineSimilarity)
