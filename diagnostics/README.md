This folder contains the diagnostic tasks for Ask Me Antying (AMA) as introduced in the main paper. The diagnostic tasks are aligned with the types of reasoning steps involved in the end-to-end AMA strategy, including question generation, text generation, extraction (from context), and selection (from listed answer choices). Given a new language model, this diagnostic may help anticipate whether AMA will provide lift. 


With a running manifest session, you can run the diagnostic as follows:

```
unzip data.zip
python run_diagnostics.py
```