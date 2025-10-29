from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall, answer_relevancy
from datasets import Dataset

# ---- Step 1: Build your dataset ----
# Each question here should correspond to a context, generated answer, and expected (ground truth) answer.

data = {
    "question": [
        "How do you think the evolution of technology has influenced the way customers interact with companies today?",
        "In what ways do you think social media and messaging apps have reshaped customer service communication?",
        "The number of social media users worldwide reached what milestone?"
    ],
    "contexts": [
        [
            "Customer service technology has come a long way over the past decades. "
            "Today’s consumers have technology at their fingertips and can choose from millions of products, "
            "but it’s the experience they have with your brand that keeps them coming back."
        ],
        [
            "The number of social media users worldwide reached 4.65 billion in 2022. "
            "Companies like Decathlon and Bupa Australia have integrated social media and messaging apps "
            "into their customer service to better reach their audiences."
        ],
        [
            "According to the ebook, the number of social media users worldwide reached 4.65 billion in 2022."
        ]
    ],
    "answers": [
        "Technology has made it easier for customers to interact with companies through digital platforms and personalized experiences.",
        "Social media and messaging apps have made customer support faster and more accessible, allowing real-time responses and engagement.",
        "It reached 4.65 billion users globally in 2022."
    ],
    "ground_truth": [
        "Technology has changed how customers engage with brands by making communication more instant and digital.",
        "Social media and messaging apps have made it possible for brands to provide direct, faster customer interactions.",
        "The ebook states that social media users reached 4.65 billion worldwide in 2022."
    ]
}

dataset = Dataset.from_dict(data)

# ---- Step 2: Run RAGAS evaluation ----
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print("\n📊 RAGAS Evaluation Results:")
print(result)
