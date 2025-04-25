## Trainer.py

This is a modified version of the Hugging Face trainer where the calculation of an additional metric in addition to the loss has been added during the training and evaluation phase: the evaluate bert f1 score.

The following have been inserted:
* a functions to import the bert score,
* some functions for extracting logits and labels, decoding them with the tokenizer and pre-processing them,
* a word management part for extracting the output responses of the considered models.

The following have been modified:
* the methods that manage training and evaluation phases, in particular, the part relating to returns with the addition of the score in addition to the loss,
* printing methods, which will no longer print just the loss but also the f1 score, both for training and evaluation.
