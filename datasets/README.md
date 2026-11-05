# Dataset preparation

Place `mosi.pkl` and/or `mosei.pkl` here. Dataset files are not bundled.

The loader expects word-aligned observations with the following structure:

```python
{
    'train': [((words, visual, acoustic), label, sample_id), ...],
    'dev':   [((words, visual, acoustic), label, sample_id), ...],
    'test':  [((words, visual, acoustic), label, sample_id), ...],
}
```

- `words`: list of tokens aligned with feature rows.
- `visual`: array of shape `(num_words, visual_dim)`.
- `acoustic`: array of shape `(num_words, audio_dim)`.
- `label`: a continuous sentiment annotation in `[-3, 3]`.

Check dataset access and redistribution permissions before publishing the data.
