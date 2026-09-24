# Attention in OpenFold3-MLX: an exploratory study

Exploratory study of the attention matrices and internal 3D visualisations of OpenFold3-MLX.

> **Status of the results below.** These numbers were produced with an earlier version of the attention extraction worker, which had two issues that have since been fixed:
> 1. it kept the **first** invocation of each layer, i.e. the first of the 4 recycling passes of the Pairformer, instead of the last one that produces the final structure;
> 2. it reduced triangle-attention cubes by averaging over **k**, the axis the softmax normalises over, which yields a constant 1/N map. This is why the triangle maps of protein B below look uniform (max weight 0.008 ≈ 1/129).
>
> The Pairformer and triangle attention figures therefore describe the initial state of the model and must be recomputed with the current worker before drawing conclusions. The diffusion transformer figures are not affected by the recycling issue. See [ARCHITECTURE.md](ARCHITECTURE.md#the-attention-worker) for how extraction works now.

## Summary

This study analyses the attention maps produced by OpenFold3-MLX to better understand how the model relates residues while predicting a structure. It combines three levels of observation:

- global prediction scores: pLDDT, pTM, gPDE, ranking score;
- residue-to-residue attention matrices extracted from several layer families;
- a 3D projection of the most connected residues onto the predicted structure.

The goal is not to prove that strong attention corresponds directly to a physical contact. It is more cautious: identify patterns in the model's internal information flow, then compare them with the 3D structure and the confidence scores.

## Research question

The main question is:

> Do the internal attention matrices of OpenFold3-MLX show distinct signatures depending on the module observed, and do these signatures help interpret the predicted structure?

Three hypotheses guide the analysis:

1. Pairformer self-attention layers combine local information with long-range relationships.
2. Triangle attention captures more indirect relational constraints between residue pairs.
3. The diffusion transformer concentrates some layers on small subsets of residues during structural denoising.

## Matrices analysed

| Family | Shape | Interpretation |
|---|---:|---|
| Pairformer self-attention | `[block, residue_i, residue_j]` | Information flow between per-residue representations, biased by the pair representation. |
| Triangle attention, starting node | `[block, residue_i, residue_j]` or sparse cube | Triangular attention oriented from the starting node. |
| Triangle attention, ending node | `[block, residue_i, residue_j]` or sparse cube | Triangular attention oriented towards the ending node. |
| Diffusion transformer token attention | `[block, residue_i, residue_j]` | Attention between tokens while the 3D structure is generated / denoised. |

Each matrix is averaged over attention heads. A strong cell means the model heavily uses information from one residue in relation to another in the observed block. It is not automatically a physical bond.

## Experimental protocol

The results below come from two local predictions run with OpenFold3-MLX and analysed with OpenFold Studio.

| Case | Length | Diffusion samples | Source |
|---|---:|---:|---|
| Protein A (GFP) | 238 residues | 8 | `var/jobs/029e9576b20b4f84a7043323acbc366d` |
| Protein B (hen egg-white lysozyme) | 129 residues | 8 | `var/jobs/4cbac540de20474c9d70008b99ff27d7` |

For each case:

1. structure prediction with OpenFold3-MLX;
2. ranking of the samples by confidence score;
3. extraction of the available attention families;
4. computation of statistics per matrix;
5. 2D visualisation as heatmaps and 3D visualisation as residue-residue links.

## Metrics

The following metrics characterise each attention family:

| Metric | Definition | Reading |
|---|---|---|
| Mean entropy | Mean entropy of the attention distributions per block | Low = concentrated attention; high = diffuse. |
| Min-entropy block | Most concentrated block | Layer that focuses information the most. |
| Max-entropy block | Most diffuse block | Layer that spreads information the most. |
| Local mass | Share of weight with `|i-j| <= 3` | Attention to sequence neighbours. |
| Long-range mass | Share of weight with `|i-j| >= 24` | Relationships between residues far apart in sequence. |
| Diagonal mass | Share of weight on `i = j` | Explicit self-attention. |
| Max weight | Largest value in the matrices | Peak intensity after averaging over heads. |

OpenFold Studio now computes these metrics for every block in the attention explorer, along with the 3D distance of the strongest links, the contact precision and attention sinks.

## Prediction results

| Case | Length | best avg pLDDT | pTM | gPDE | ranking score |
|---|---:|---:|---:|---:|---:|
| Protein A | 238 | 89.32 | 0.869 | 0.460 | 0.182 |
| Protein B | 129 | 87.27 | 0.769 | 0.525 | 0.154 |

Both predictions are in a usable confidence range, with a mean pLDDT around 87–89. Protein A has a more confident global topology according to pTM, while protein B is more moderate. For protein A, the per-residue analysis shows a single low-confidence region: the C-terminal tail (residues 231–238, mean pLDDT 33).

## Attention matrix results

### Protein A, 238 residues

| Family | Blocks | Mean entropy | Min entropy | Max entropy | Local mass | Long-range mass | Diagonal mass | Max weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pairformer self-attention | 48 | 4.58 | 3.79 block 0 | 5.09 block 14 | 0.203 | 0.597 | 0.023 | 0.277 |
| Triangle starting | 48 | 3.49 | 2.12 block 19 | 4.82 block 4 | 0.140 | 0.706 | 0.001 | 0.715 |
| Triangle ending | 48 | 3.50 | 1.91 block 8 | 4.52 block 5 | 0.220 | 0.598 | 0.001 | 0.729 |
| Diffusion token | 24 | 4.57 | 3.06 block 8 | 5.20 block 1 | 0.257 | 0.577 | 0.052 | 0.408 |

Observations:

- Triangle attentions are the most concentrated: mean entropy around 3.5, with maximum weights above 0.7.
- Triangle starting puts about 70.6% of its mass on long-range relationships, suggesting heavy use of non-local constraints.
- The diffusion transformer is more diffuse on average, but block 8 becomes strongly concentrated.
- Pairformer self-attention combines a noticeable local mass with a strong long-range component.

Strongest long-range links in the most concentrated blocks:

| Family | Block | Strong links |
|---|---:|---|
| Pairformer self-attention | 0 | 38-89 `w=0.129`, 203-89 `w=0.129`, 198-89 `w=0.129` |
| Triangle starting | 19 | 102-1 `w=0.606`, 173-1 `w=0.584`, 212-1 `w=0.570` |
| Triangle ending | 8 | 190-2 `w=0.543`, 192-2 `w=0.500`, 28-2 `w=0.487` |
| Diffusion token | 8 | 231-77 `w=0.201`, 227-200 `w=0.192`, 228-200 `w=0.163` |

Note that many strong links share one endpoint (89, 1, 2, 200): this is the signature of **attention sinks**, residues that receive attention from almost everyone. They should be interpreted with care (see Interpretation).

### Protein B, 129 residues

| Family | Blocks | Mean entropy | Min entropy | Max entropy | Local mass | Long-range mass | Diagonal mass | Max weight |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pairformer self-attention | 48 | 4.00 | 3.33 block 0 | 4.55 block 14 | 0.215 | 0.479 | 0.025 | 0.296 |
| Triangle starting | 48 | 4.86 | 4.86 block 33 | 4.86 block 3 | 0.054 | 0.669 | 0.008 | 0.008 |
| Triangle ending | 48 | 4.86 | 4.86 block 0 | 4.86 block 25 | 0.054 | 0.669 | 0.008 | 0.008 |
| Diffusion token | 24 | 4.19 | 2.93 block 8 | 4.65 block 1 | 0.280 | 0.439 | 0.058 | 0.403 |

Observations:

- The diffusion transformer again shows a clear concentration at block 8, with a maximum weight around 0.403.
- Pairformer self-attention remains more structured than triangle attention in this extraction.
- The triangle attentions of protein B are perfectly uniform (max weight 0.008 ≈ 1/129, entropy = log 129 ≈ 4.86). This was an artefact of the old reduction over the softmax axis (see the status note at the top), not a property of the model. Recomputed with the current reduction from the saved cubes, these maps are clearly structured (max weight ≈ 0.51, mean entropy ≈ 4.0 for triangle starting).

Strongest long-range links in the most concentrated blocks:

| Family | Block | Strong links |
|---|---:|---|
| Pairformer self-attention | 0 | 13-79 `w=0.132`, 129-79 `w=0.130`, 12-79 `w=0.130` |
| Triangle starting | 33 | 24-112 `w=0.008`, 98-47 `w=0.008`, 54-120 `w=0.008` |
| Triangle ending | 0 | 24-102 `w=0.008`, 71-37 `w=0.008`, 71-44 `w=0.008` |
| Diffusion token | 8 | 6-127 `w=0.286`, 127-6 `w=0.261`, 100-21 `w=0.214` |

## Interpretation

The results support a reading in three regimes:

1. **Local regime**: visible as bands near the diagonal, linked to sequence neighbours and secondary-structure motifs.
2. **Long-range regime**: visible as off-diagonal points and 3D links, potentially relevant for the global fold.
3. **Diffuse regime**: visible as high entropy, indicating a layer that spreads information over many residues.

In both cases, block 8 of the diffusion transformer stands out as strongly concentrated. This recurrence deserves a systematic analysis on more sequences.

A fourth pattern appears clearly in the explorer: **attention sinks**. In Pairformer block 0 of human hemoglobin α (142 residues, extracted with the current worker), residue 139 (S) receives 13% of all attention, 19 times its uniform share, and none of the 10 strongest long-range links it takes part in is a 3D contact (Cα distances of 14–31 Å). Links to sinks say little about structure and should be filtered out before looking for contact-like attention.

## Visualisations

OpenFold Studio uses complementary views:

- **2D heatmap**: each cell `(i, j)` is the mean attention weight between two residues.
- **3D structure**: residues strongly linked to the selected residue are coloured and connected by lines on the predicted structure.
- **Block insights**: focus, self / local / long-range shares, attention by sequence separation, strongest long-range links with their 3D distance, contact precision and sinks.
- **Focus across layers**: entropy of every block, to find the sharpest layers.

Colour reading:

- light blue: weak or moderate attention;
- yellow: intermediate attention;
- orange: strong attention;
- dark blue: selected residue.

Labels show the residue number and its amino-acid letter, for example `119 (R)`.

## Limitations

This study is exploratory and must be interpreted with caution.

- Attention weights are not physical distances.
- Matrices are averaged over heads, which can hide per-head specialisation.
- The results only cover two local sequences.
- Triangle attention is sensitive to how it is extracted and reduced (see the status note).
- Local jobs and raw data should not be published as is if they contain private sequences.

## Future work

To turn this analysis into a more robust study:

1. Re-run all extractions with the current worker and update the tables above.
2. Add at least 20–50 proteins of varied lengths and topologies.
3. Compare attention links with 3D contacts derived from Cα–Cα distances. The explorer now reports this contact precision per block; it should be aggregated over proteins and layers.
4. Measure precision, recall and enrichment of attention links as contact predictors, after removing attention sinks.
5. Analyse heads separately instead of only their average.
6. Add exportable figures: heatmaps, entropy distributions, residue-residue graphs.
7. Publish a reproducible script `scripts/analyze_attention.py`.

## Reproducibility

Start the app as described in the [README](../README.md#installation), then:

1. Run a prediction.
2. Open the completed job.
3. Select the attention families.
4. Start the analysis.
5. Explore the heatmaps, the 3D structure and the block insights.

Expected outputs per run:

```text
meta.json
pairformer_self_attn.npz
triangle_start.npz
triangle_end.npz
diffusion_token.npz
*_cube.npz
```

## Positioning

OpenFold Studio is not a standalone biological validation tool. It is an interpretability environment to observe how OpenFold3-MLX distributes information between residues during prediction. Its results must be cross-checked with confidence scores, structural distances, biological annotations and, when possible, experimental structures.
