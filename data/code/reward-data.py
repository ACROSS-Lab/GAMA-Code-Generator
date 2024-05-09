import json
import random as rd
import csv
import json
from tqdm import tqdm


def read_json(input):
    base_question = []
    base_answer = []
    with open(input, 'r') as file:
        data = json.load(file)

    for item in data:
        base_question.append(item["question"])
        base_answer.append(item["answer"])

    return base_answer,base_question

def crossover(split_percentage1, split_percentage2):
    split_percentage1 = rd.uniform(0.3, 0.5)
    split_percentage2 = rd.uniform(0.1, 0.4)
    # Crossover function for split percentages
    return (split_percentage1 + split_percentage2) / 2

def mutate(split_percentage, mutation_rate):
    # Mutation function for split percentages
    split_percentage = rd.uniform(0.4, 0.6)
    mutation = rd.uniform(-mutation_rate, mutation_rate)
    return max(0.0, min(1.0, split_percentage + mutation))

def merge_paragraphs(part1_lines, part2_lines, split_percentage):
    split_index = int(len(part1_lines) * split_percentage)
    start_with_part1 = rd.choice([True, False])

    merged_paragraph = ''
    if start_with_part1:
        merged_paragraph += '\n'.join(part1_lines[:split_index])
        merged_paragraph += '\n'.join(part2_lines[split_index:])
    else:
        merged_paragraph += '\n'.join(part2_lines[:split_index])
        merged_paragraph += '\n'.join(part1_lines[split_index:])

    return merged_paragraph

def fitness(merged_paragraph, target_text):
    # Define a simple fitness function, for example, based on how close the merged paragraph is to the target text
    return sum(1 for a, b in zip(merged_paragraph, target_text) if a == b)

def genetic_algorithm(base_answer, generations, base_question):
    population_size = 10
    mutation_rate = 0.1
    new_answers = []

    for loop in tqdm(range(generations)):
        new_population = []

        for _ in range(population_size):
            parent1 = rd.choice(base_answer)
            parent2 = rd.choice(base_answer)

            idx1 = base_answer.index(parent1)
            idx2 = base_answer.index(parent2)

            question1 = base_question[idx1]
            question2 = base_question[idx2]

            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        individual1 = [{'split_percentage': ind, 'fitness': fitness(merge_paragraphs(parent1.split('\n'), parent2.split('\n'), ind), parent1)} for ind in new_population]
        individual2 = [{'split_percentage': ind, 'fitness': fitness(merge_paragraphs(parent1.split('\n'), parent2.split('\n'), ind), parent2)} for ind in new_population]
        total_individuals = individual1 + individual2
        individuals = sorted(total_individuals, key=lambda x: x['fitness'], reverse=True)

        best_individuals_batched = []

        for i in range(0, len(individuals), 3):
            batch = []
            for j in range(3):
                if i + j < len(individuals):
                    best_individual = individuals[i + j]
                    new_answer = format(merge_paragraphs(parent1.split('\n'), parent2.split('\n'), best_individual['split_percentage']))
                    batch.append(new_answer)
            best_individuals_batched.append(batch)

        for i, batch in enumerate(best_individuals_batched):
            # Extract parents for the current batch
            question_batch = [question1, question2] * (len(batch) // 2)  # Duplicate parents if needed
            # Zip child and its parents into tuples
            children_with_question = list(zip(question_batch, batch))
            new_answers.extend(children_with_question)

    new_answers = list(new_answers)
    print(new_answers)
    return new_answers


def create_reward_data(new_answer, base_question, base_answer):
    reward_qt = []
    reward_ans = []

    for item in new_answer:
        item = list(item)

    

    
    #print(type(new_answer))
    #print(len(reward_ans))
    #print(len(reward_qt))
        
    grouped_values = {}
    for question, answer in new_answer:
        if question in grouped_values:
            grouped_values[question].append(answer)  # Initialize an empty list for the question if not exists
        else:
            grouped_values[question] = [answer]
    
    #grouped_values = list(grouped_values)
    #print(grouped_values.get(first_question))


    counts_by_questions = {}
    # Count values grouped by 'a', 'b', 'c', etc.
    for question, answer in new_answer:
        if question in counts_by_questions:
            counts_by_questions[question] += 1
        else:
            counts_by_questions[question] = 1

    # Because after Genetic Algorithm, we have to many answer for each question, so we need to choose random from it 3 answers
    values_to_choose = {}
    for group, counts in counts_by_questions.items():
        values_to_choose[group] = rd.sample(grouped_values[group], min(1, counts))
    #values_to_choose = list(values_to_choose)
    
    # Combine the chosen values with base answer (wanted answer) before write to csv file
    # Make base list
    base_list = []
    for id in range(len(base_question)): 
        sample = list(zip(base_question[id], base_answer[id])) # Combine and make item the list
        # Convert item to list
        base_list.append(sample)
  
    return base_list, values_to_choose
        

def combine_prompts(path_link):
    new_prompts = json.load(open(f'{path_link}/temp.json'))
    base_prompts_list = json.load(open(f'{path_link}/total.json'))

    print(len(base_prompts_list))
    new_prompts_list = new_prompts['data']['prompts']
    merged_prompts = base_prompts_list + new_prompts_list


    with open(f'{path_link}/new-reward.json', 'w', encoding = 'utf-8') as jf:
	    jf.write(json.dumps(merged_prompts, indent = 4))


def append_system_to_prompt(base_list, values_to_choose, path_link):


    combined_list = []
    for item in base_list:
        print(item)
        if item['question'] in values_to_choose:
            combined_list.append(
                ["You are an AI assistant. You will be given a task generate a specific GAML code snippet."]                 
                + item 
                + values_to_choose[item['question']])
        else:
            combined_list.append(
                ["You are an AI assistant. You will be given a task generate a specific GAML code snippet."]
                + item 
                + [''])

    
    # Write data to CSV
    with open(f"{path_link}/new-reward.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        header = ['system'] + ['question'] + ['chosen'] + ['rejected']
        writer.writerow(header)  # Write header

        # Write randomly chosen values to CSV
        for item in combined_list:
            writer.writerow(item)
    
    
    
if __name__ == "__main__":
    generations = 10000
    path_link = "/home/phuong-anh/gama/data/train/json"
    name = input("Enter filename: ")
    input_path = f"{path_link}/{name}"
    base_answer, base_question = read_json(input_path)
    # print(len(base_question))
    #create_new_answer(pairs_of_answer)
    new_answer = genetic_algorithm(base_answer, generations, base_question)
    base_list, values_to_choose = create_reward_data(new_answer, base_question, base_answer)
    
    #append_system_to_prompt(base_list, values_to_choose, path_link)

    #combine_prompts(path_link)

