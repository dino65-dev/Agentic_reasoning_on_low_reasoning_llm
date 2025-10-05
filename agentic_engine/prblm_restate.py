from problem_restatement.problem_restatement import AdvancedProblemRestater
restater = AdvancedProblemRestater(enable_spacy=True)
problems = ["HI,how are you!"]
results = restater.batch_restate_parallel(problems, max_workers=4)
print(results)
