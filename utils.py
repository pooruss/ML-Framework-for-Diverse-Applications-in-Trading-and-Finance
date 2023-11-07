def parse_solution(solution):
    if "<CODE>" in solution:
        start_index = solution.find("<CODE>")+len("<CODE>")
    elif "```python" in solution:
        start_index = solution.find("```python")+len("```python")
    else:
        start_index = 0
    if "</CODE>" in solution:
        end_index = solution.find("</CODE>")
    else:
        end_index = -1
    solution = solution[start_index: end_index]
    while "```" in solution:
        solution = solution.replace("```", "")
    return solution
