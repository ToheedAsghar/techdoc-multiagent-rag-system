from backend.agents.graph import run_graph

result = run_graph("Write a paragraph on lahore.")
print(result['synthesized_answer'])