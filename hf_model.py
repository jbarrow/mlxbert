from transformers import AutoModel, AutoTokenizer


if __name__ == "__main__":
    batch = [
        "This is an example of BERT working on MLX.",
        "A second string",
        "This is another string.",
    ]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    torch_model = AutoModel.from_pretrained("bert-base-uncased")
    torch_tokens = tokenizer(batch, return_tensors="pt", padding=True)
    torch_forward = torch_model(**torch_tokens)
    torch_output = torch_forward.last_hidden_state.detach().numpy()
    torch_pooled = torch_forward.pooler_output.detach().numpy()

    print("\n HF BERT:")
    print(torch_output)
    print("\n\n HF Pooled:")
    print(torch_pooled[0, :20])