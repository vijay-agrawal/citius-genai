##
# Template for implementing evaluations for RAG applications
#
test_data = [
    {
       "prompt": "who has java skills?",
       "ground_truth": "Sujata, Rajesh Chaudhary"
    }
    # Add more test items with different queries & ground truths
]

# --------------------------------------------------------------------------
# Evaluate Over the Test Dataset
# --------------------------------------------------------------------------

predictions = []
references = []

for item in test_data:
    query = item["prompt"]
    ground_truth = item["ground_truth"]

    # 4.1 Retrieve top-k chunks for this query
    query_emb = embedding_model.encode(query).reshape(1, -1)
    D, I = index.search(query_emb, k=3)  # retrieve top-3 docs
    retrieved_indices = I[0]

    # Collect the relevant text from these retrieved docs
    retrieved_snippets = [resumes[idx]["text"] for idx in retrieved_indices]

    # 4.2 Generate the final answer by passing the query and the snippets to the LLM
    answer = generate_answer_from_llm(query, retrieved_snippets)

    predictions.append(answer)
    references.append(ground_truth)

# --------------------------------------------------------------------------
# Step 5: Compute BERTScore
# --------------------------------------------------------------------------
# BERTScore compares predicted texts with references in a more semantic way.

P, R, F1 = score(predictions, references, model_type='bert-base-uncased')
avg_precision = P.mean().item()
avg_recall = R.mean().item()
avg_f1 = F1.mean().item()

print("BERTScore results:")
print("Precision:", avg_precision)
print("Recall:", avg_recall)
print("F1:", avg_f1)

# For debugging, we can also print each pair of (prediction, reference) with its BERT F1
for i, (pred, ref) in enumerate(zip(predictions, references)):
    print(f"\nQuery: {test_data[i]['prompt']}")
    print("Prediction:", pred)
    print("Ground Truth:", ref)
    print("BERTScore (F1) =", F1[i].item())