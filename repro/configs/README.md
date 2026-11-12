# Dataset recipes

Each YAML fixes full gallery dimensions, PQ4 FastScan layout, partition/PQ samples,
teacher and student architectures, optimizer settings and Recall@100 targets.
Use `mhcr-pipeline --config <yaml> --inputs <input-yaml> --out <new-directory>`.
Input paths belong in the separate input YAML, not in dataset recipes.

`student.budget_range: null` requests training-only budget derivation after the
new index has been measured. The output `resolved_config.yaml` contains the actual
range. `published_training` records reference query counts and previously selected
ranges; those ranges must not be reused with unrelated costs or budget units.

The teacher uses rank-exponential weights with gamma20. Student GT100 coverage
is unweighted; the student also receives the teacher distribution as a distillation
term. DataComp retains its millisecond budget and12-step gate; the other recipes
use fractions of gallery entries and the32-step implicit gate.

These are complete CPU reproduction recipes, not serialized historical states.
New centroids and PCA bases need not be bitwise identical to historical GPU training.
The YFCC recipe trains the teacher for all20 configured epochs, rather than an
elapsed-time cutoff used during parameter exploration. Keep those execution choices
in mind when comparing a fresh run with a selected published model.

For a small engineering test, copy a recipe and explicitly change its dimensions,
sample counts and training sizes. Such a run validates the pipeline, not the full
published experiment. Never silently shorten the gallery or change returned top-k.
