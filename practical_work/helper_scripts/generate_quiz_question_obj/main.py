def generate_quiz_questions(input_file, output_file, start_id):
    # Read the input file
    with open(input_file, 'r') as infile:
        content = infile.read()

    question_blocks = [block.strip() for block in content.split('\n\n') if block.strip()]

    # Prepare the output string
    output = "[\n"
    current_id = start_id

    for block in question_blocks:
        # Split the block into lines
        lines = [line.strip() for line in block.split('\n') if line.strip()]

        if not lines:
            continue

        # First line contains question and difficulty
        question_line = lines[0]
        difficulty = lines[1][1:len(lines[1]) - 1]

        # Extract difficulty and clean up question text
        question = question_line

        # Get answers (remaining lines)
        answers = []
        for answer_line in lines[2:]:
            # Skip any lines that contain difficulty markers
            if any(f'({diff})' in answer_line.lower() for diff in ['easy', 'medium', 'hard']):
                continue

            if '(Correct)' in answer_line:
                answer_text = answer_line.replace('(Correct)', '').strip()
                answers.append((answer_text, True))
            else:
                answers.append((answer_line.strip(), False))

        # Format and add the question
        output += "  QuizQuestion(\n"
        output += f"    id: {current_id},\n"
        output += f'    question: "{question}",\n'
        output += f'    difficulty: "{difficulty}",\n'
        output += "    allAnswers: [\n"

        # Add answers
        for answer in answers:
            output += f'      QuizAnswer.text(quizQuestionId: {current_id}, text: "{answer[0]}", isCorrect: {str(answer[1]).lower()}),\n'

        output += "    ],\n"
        output += "  ),\n"

        current_id += 1

    # Close the list
    output = output.rstrip(",\n") + "\n]"

    # Write the output to the output file
    with open(output_file, 'w') as outfile:
        outfile.write(output)


# Example usage
if __name__ == "__main__":
    input_file = 'input_questions.txt'
    output_file = 'output.txt'
    start_id = 23

    generate_quiz_questions(input_file, output_file, start_id)