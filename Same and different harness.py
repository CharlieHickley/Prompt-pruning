from leven import normalised_lev
from levenshtein import levenshtein_distance_words as lev
from cosine import get_cosineSimilarity as cosine
from ncd import ncd
import matplotlib.pyplot as plt
from yaml import safe_load_all, YAMLError
import openai
from pandas import DataFrame
from statistics import median

openai.api_key = ""

K = 10
MODEL = "gpt-4o-mini"
DIFFERENT_PROMPT = "I want you to act as a mathematical history teacher and provide information about the historical development of mathematical concepts and the contributions of different mathematicians. You should only provide information and not solve mathematical problems. Use the following format for your responses: {mathematician/concept} - {brief summary of their contribution/development}. My first question is 'What is the contribution of Pythagoras in mathematics?"
THRESHOLD = None


def read_prompts(path="./data/prompt50.yaml"):
    """
    Reads the prompt file and returns the prompts in list

    Args:
        path (str, optional): The path to prompt file. Defaults to './data/prompt50.yaml'.

    Returns:
        list: list of prompts
        exit(1): if an error occured
    """

    try:
        with open(path, "r") as file:

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


def get_response(prompt, act, model=MODEL):
    output = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Act like you are" + act},
            {"role": "user", "content": prompt},
        ],
    )
    response = output.choices[0].message.content
    return response


def get_responses(prompt, act, model=MODEL):
    """
    Call openai api to get a list of random choices

    Args:
        prompt (str): The content of the prompt
        act (str): What the model should act as
        model (str, optional): The model from openAI API. Defaults to "gpt-4o-mini".

    Returns:
        list: list of predictions
    """
    output = openai.ChatCompletion.create(
        model=model,
        n=K + 1,
        messages=[
            {"role": "system", "content": "Act like you are" + act},
            {"role": "user", "content": prompt},
        ],
    )
    responses = [choice.message.content for choice in output.choices]
    return responses


def generate_set_of_similarity(
    prompt, act, similarity_measure, different_prompt=DIFFERENT_PROMPT, model=MODEL
):
    """
    Generate a sample set of similarity under a meassure

    Args:
        prompt (str): The content of the prompt
        act (str): What the model should act as
        different_prompt (str): The content of the different prompt
        similarity_meassure (function): The similarity meassure function
        model (str, optional): The model from OpenAI API. Defaults to "gpt-4o-mini".

    Returns:
        tuple: a tuple of two lists of similarity (one for different prompt and one for same prompt)
        None: if error occured
    """
    try:
        reponses = get_responses(prompt, act, model)
    except openai.error.OpenAIError as exc:
        print(exc)
        return None

    try:
        different_response = get_response(different_prompt, act, model)
    except openai.error.OpenAIError as exc:
        print(exc)
        return None

    subset_1 = reponses[0]
    subset_2 = reponses[1:]
    subset_3 = different_response

    set_of_similarity = (
        [similarity_measure(subset_3, subset_2i) for subset_2i in subset_2],
        [similarity_measure(subset_1, subset_2i) for subset_2i in subset_2],
    )
    print(f"{act}: {str(set_of_similarity)}\n")
    return set_of_similarity


def plot(data):
    """Plot the boxplot

    Args:
        data : list of tuples of two lists (one for different prompt and one for same prompt)
    """
    fig, ax = plt.subplots()
    for i, (different, same) in enumerate(data):
        pos1 = i * 2 + 0.8
        pos2 = i * 2 + 0.8

        bp1 = ax.boxplot(
            different,
            positions=[pos1],
            widths=0.8,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", color="blue"),
            capprops=dict(color="blue"),
            whiskerprops=dict(color="blue"),
            flierprops=dict(color="blue", markeredgecolor="blue"),
            medianprops=dict(color="blue"),
        )
        bp2 = ax.boxplot(
            same,
            positions=[pos2],
            widths=0.8,
            patch_artist=True,
            boxprops=dict(facecolor="lightgreen", color="green"),
            capprops=dict(color="green"),
            whiskerprops=dict(color="green"),
            flierprops=dict(color="green", markeredgecolor="green"),
            medianprops=dict(color="green"),
        )
    ax.set_xticks([i * 2 + 0.8 for i in range(len(data))])
    ax.set_xticklabels([f"{i+1}" for i in range(len(data))])
    ax.legend(
        [bp1["boxes"][0], bp2["boxes"][0]],
        ["Different prompt", "Same prompt"],
        loc="upper right",
    )

    plt.title("Sample sets for word granular levenshtein distance")
    plt.xlabel("Sample")
    plt.ylabel("Similarity")
    plt.grid(axis="both")
    plt.show()


def any_result_matches(
    prompt, act, similarity_meassure, model=MODEL, threshold=THRESHOLD
):
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


def print_stats_median(data):
    """Print the median stats for different and same prompts

    Args:
        data: list of tuples of two lists (one for different prompt and one for same prompt)
    """
    medianList_different = []
    medianList_same = []
    for different, same in data:
        medianList_different.append(median(different))
        medianList_same.append(median(same))

    data_different_prompt = {"median Stats for different prompts": medianList_different}
    data_same_prompt = {"median Stats for same prompts": medianList_same}
    stats_different = DataFrame(data_different_prompt).describe()
    stats_same = DataFrame(data_same_prompt).describe()

    print(f"{stats_different}\n\n{stats_same}")


def main(similarity_meassure):

    prompts = read_prompts()
    data = [
        generate_set_of_similarity(prompt["prompt"], prompt["act"], similarity_meassure)
        for prompt in prompts
    ]

    print_stats_median(data)

    plot(data)


if __name__ == "__main__":
    main(lev)
