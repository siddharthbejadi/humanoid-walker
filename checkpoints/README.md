# Checkpoints

`student_realistic_12k.pt` (112 MB, over GitHub's 100 MB file limit) is hosted on Hugging Face: https://huggingface.co/siddharthbejadi/humanoid-walker-student (currently private).

    hf download siddharthbejadi/humanoid-walker-student student_realistic_12k.pt --local-dir checkpoints
    sha256sum checkpoints/student_realistic_12k.pt
    # 1d1af540214b7be6f201eed97bd47c0adca665b309b230968688cf1a5db806ce

`HF_MODEL_CARD.md` is the model card published as the Hugging Face repo's README. You can also rebuild the weights from scratch: see "Reproduce the training" in the top-level README.
