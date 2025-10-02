from datasets import load_dataset, concatenate_datasets
import pyarrow.parquet as pq
import os

def truthfulfn(batch):

  for q, correct_a, incorrect_a in zip(batch['Question'], batch['Correct Answers'], batch['Incorrect Answers']):
    i_answers = incorrect_a.split("; ")
    c_answers = correct_a.split("; ")
    total_len = len(i_answers) + len(c_answers)

    return {
        "Knowledge": ["" for i in range(total_len)],
        "Answer": i_answers + c_answers,
        "Question": [q for i in range(total_len)],
        "Hallucination": ["yes" for i in range(len(i_answers))] + ["no" for i in range(len(c_answers))]
    }

def halufn(batch):
  for question, knowledge, c_answer, i_answer in zip(batch['question'], batch['knowledge'], batch['right_answer'], batch['hallucinated_answer']):

    return {
        'Knowledge': [knowledge, knowledge],
        'Question': [question, question],
        'Answer': [c_answer, i_answer],
        "Hallucination":["no", "yes"],
    }

if __name__ == "__main__":

    truthfulqa = load_dataset("domenicrosati/TruthfulQA")
    
    processed_truthfulqa = truthfulqa.map(
        truthfulfn,
        batched=True,
        batch_size=1,
        remove_columns = ['Type', 'Category', 'Best Answer', 'Source', 'Correct Answers', 'Incorrect Answers']
    )
    
    halueval = load_dataset("pminervini/HaluEval", "qa")

    processed_halueval = halueval.map(
        halufn,
        batched=True,
        batch_size=1,
        remove_columns=['knowledge', 'question', 'right_answer', 'hallucinated_answer']
    )

    dataset = concatenate_datasets([processed_truthfulqa['train'], processed_halueval['data']])
    print(os.getcwd())
    dataset.to_parquet("data/data.parquet")