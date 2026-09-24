# OPENFOLD_PROJECT

Etude exploratoire des matrices d'attention et des visualisations 3D internes d'OpenFold3-MLX.

## Resume

OPENFOLD_PROJECT analyse les cartes d'attention produites par OpenFold3-MLX afin de mieux comprendre comment le modele relie les residus pendant la prediction de structure. Le projet combine trois niveaux d'observation:

- les scores globaux de prediction: pLDDT, pTM, gPDE, score de classement;
- les matrices d'attention residu-residu extraites de plusieurs familles de couches;
- la projection 3D des residus les plus connectes sur la structure predite.

L'objectif n'est pas de prouver qu'une attention forte correspond directement a un contact physique. L'objectif est plus prudent: identifier les motifs d'information interne du modele, puis les comparer a la structure 3D et aux scores de confiance.

## Question De Recherche

La question principale est:

> Les matrices d'attention internes d'OpenFold3-MLX presentent-elles des signatures distinctes selon le module observe, et ces signatures aident-elles a interpreter la structure predite?

Trois hypotheses guident l'analyse:

1. Les couches Pairformer self-attention combinent information locale et relations longue distance.
2. Les triangle attentions capturent davantage de contraintes relationnelles indirectes entre paires de residus.
3. Le Diffusion Transformer concentre certaines couches sur des sous-ensembles reduits de residus pendant le denoising structural.

## Matrices Analysees

| Famille | Forme | Interpretation |
|---|---:|---|
| Pairformer self-attention | `[bloc, residu_i, residu_j]` | Flux d'information entre representations par residu, biaise par la representation de paires. |
| Triangle attention starting node | `[bloc, residu_i, residu_j]` ou cube sparse | Attention triangulaire orientee depuis le noeud de depart. |
| Triangle attention ending node | `[bloc, residu_i, residu_j]` ou cube sparse | Attention triangulaire orientee vers le noeud d'arrivee. |
| Diffusion Transformer token attention | `[bloc, residu_i, residu_j]` | Attention entre tokens pendant la generation/denoising de la structure 3D. |

Chaque matrice est moyennee sur les tetes d'attention. Une case forte signifie que le modele utilise fortement l'information d'un residu en relation avec un autre dans le bloc observe. Ce n'est pas automatiquement une liaison physique.

## Protocole Experimental

Les resultats ci-dessous proviennent de deux predictions locales executees avec OpenFold3-MLX et analysees avec OpenFold Studio.

| Cas | Longueur | Echantillons de diffusion | Source |
|---|---:|---:|---|
| Proteine A | 238 residus | 8 | `var/jobs/029e9576b20b4f84a7043323acbc366d` |
| Proteine B | 129 residus | 8 | `var/jobs/4cbac540de20474c9d70008b99ff27d7` |

Pour chaque cas:

1. prediction de structure avec OpenFold3-MLX;
2. classement des echantillons par score de confiance;
3. extraction des familles d'attention disponibles;
4. calcul des statistiques par matrice;
5. visualisation 2D par heatmap et 3D par liens residu-residu.

## Metriques

Les metriques suivantes sont utilisees pour caracteriser chaque famille d'attention:

| Metrique | Definition | Lecture |
|---|---|---|
| Entropie moyenne | Entropie moyenne des distributions d'attention par bloc | Basse = attention concentree; haute = attention diffuse. |
| Bloc min entropy | Bloc le plus concentre | Couche qui focalise le plus l'information. |
| Bloc max entropy | Bloc le plus diffus | Couche qui distribue le plus largement l'information. |
| Masse locale | Fraction de poids avec `|i-j| <= 3` | Attention proche dans la sequence. |
| Masse longue distance | Fraction de poids avec `|i-j| >= 24` | Relations entre residus eloignes dans la sequence. |
| Masse diagonale | Fraction de poids sur `i = j` | Auto-attention explicite. |
| Poids maximal | Plus grande valeur dans les matrices | Intensite maximale observee apres moyenne sur tetes. |

## Resultats De Prediction

| Cas | Longueur | meilleur avg pLDDT | pTM | gPDE | score de classement |
|---|---:|---:|---:|---:|---:|
| Proteine A | 238 | 89.32 | 0.869 | 0.460 | 0.182 |
| Proteine B | 129 | 87.27 | 0.769 | 0.525 | 0.154 |

Les deux predictions sont dans une zone de confiance exploitable: pLDDT moyen autour de 87-89. La proteine A presente une topologie globale plus confiante d'apres pTM, tandis que la proteine B est plus moderee.

## Resultats Des Matrices D'Attention

### Proteine A, 238 residus

| Famille | Blocs | Entropie moyenne | Min entropy | Max entropy | Masse locale | Masse longue distance | Masse diagonale | Poids max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pairformer self-attention | 48 | 4.58 | 3.79 bloc 0 | 5.09 bloc 14 | 0.203 | 0.597 | 0.023 | 0.277 |
| Triangle starting | 48 | 3.49 | 2.12 bloc 19 | 4.82 bloc 4 | 0.140 | 0.706 | 0.001 | 0.715 |
| Triangle ending | 48 | 3.50 | 1.91 bloc 8 | 4.52 bloc 5 | 0.220 | 0.598 | 0.001 | 0.729 |
| Diffusion token | 24 | 4.57 | 3.06 bloc 8 | 5.20 bloc 1 | 0.257 | 0.577 | 0.052 | 0.408 |

Observations:

- Les triangle attentions sont les plus concentrees: entropie moyenne autour de 3.5, avec des poids maximaux superieurs a 0.7.
- Triangle starting consacre environ 70.6% de sa masse a des relations longue distance, ce qui suggere une forte utilisation de contraintes non locales.
- Le Diffusion Transformer est plus diffuse en moyenne, mais le bloc 8 devient fortement concentre.
- La self-attention Pairformer combine une masse locale notable avec une forte composante longue distance.

Liens longue distance les plus forts dans les blocs les plus concentres:

| Famille | Bloc | Liens forts |
|---|---:|---|
| Pairformer self-attention | 0 | 38-89 `w=0.129`, 203-89 `w=0.129`, 198-89 `w=0.129` |
| Triangle starting | 19 | 102-1 `w=0.606`, 173-1 `w=0.584`, 212-1 `w=0.570` |
| Triangle ending | 8 | 190-2 `w=0.543`, 192-2 `w=0.500`, 28-2 `w=0.487` |
| Diffusion token | 8 | 231-77 `w=0.201`, 227-200 `w=0.192`, 228-200 `w=0.163` |

### Proteine B, 129 residus

| Famille | Blocs | Entropie moyenne | Min entropy | Max entropy | Masse locale | Masse longue distance | Masse diagonale | Poids max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pairformer self-attention | 48 | 4.00 | 3.33 bloc 0 | 4.55 bloc 14 | 0.215 | 0.479 | 0.025 | 0.296 |
| Triangle starting | 48 | 4.86 | 4.86 bloc 33 | 4.86 bloc 3 | 0.054 | 0.669 | 0.008 | 0.008 |
| Triangle ending | 48 | 4.86 | 4.86 bloc 0 | 4.86 bloc 25 | 0.054 | 0.669 | 0.008 | 0.008 |
| Diffusion token | 24 | 4.19 | 2.93 bloc 8 | 4.65 bloc 1 | 0.280 | 0.439 | 0.058 | 0.403 |

Observations:

- Le Diffusion Transformer montre encore une concentration nette au bloc 8, avec un poids maximal autour de 0.403.
- Pairformer self-attention reste plus structuree que les triangle attentions dans cette extraction.
- Les triangle attentions de la proteine B sont quasi uniformes dans les matrices reduites observees: poids max autour de 0.008 et entropie constante. Cela peut indiquer soit une vraie diffusion du signal, soit une limite de l'extraction/moyennage pour ce cas.

Liens longue distance les plus forts dans les blocs les plus concentres:

| Famille | Bloc | Liens forts |
|---|---:|---|
| Pairformer self-attention | 0 | 13-79 `w=0.132`, 129-79 `w=0.130`, 12-79 `w=0.130` |
| Triangle starting | 33 | 24-112 `w=0.008`, 98-47 `w=0.008`, 54-120 `w=0.008` |
| Triangle ending | 0 | 24-102 `w=0.008`, 71-37 `w=0.008`, 71-44 `w=0.008` |
| Diffusion token | 8 | 6-127 `w=0.286`, 127-6 `w=0.261`, 100-21 `w=0.214` |

## Interpretation

Les resultats soutiennent une lecture en trois regimes:

1. **Regime local**: visible par les bandes proches de la diagonale, lie aux voisins de sequence et aux motifs secondaires.
2. **Regime longue distance**: visible par des points hors diagonale et par les liens 3D, potentiellement pertinent pour le repliement global.
3. **Regime diffus**: visible par une entropie elevee, indiquant une couche qui repartit l'information sur beaucoup de residus.

Dans les deux cas, le bloc 8 du Diffusion Transformer apparait comme un bloc fortement concentre. Cette recurrence merite une analyse systematique sur plus de sequences.

La proteine A montre des triangle attentions tres marquees, avec plusieurs liens longue distance forts. La proteine B montre au contraire des triangle attentions beaucoup plus uniformes dans les matrices extraites, tandis que Pairformer et Diffusion restent informatifs.

## Visualisations

OPENFOLD_PROJECT utilise deux vues complementaires:

- **Heatmap 2D**: chaque case `(i, j)` represente le poids d'attention moyen entre deux residus.
- **Structure 3D**: les residus fortement relies au residu selectionne sont colores et connectes par des traits dans la structure predite.

Lecture des couleurs:

- bleu: attention faible ou moderee;
- jaune: attention intermediaire;
- orange/rouge: attention forte;
- bleu fonce: residu selectionne.

Les etiquettes affichent le numero du residu et sa lettre d'acide amine, par exemple `119 (R)`.

## Limites

Cette etude est exploratoire et doit etre interpretee avec prudence.

- Les poids d'attention ne sont pas des distances physiques.
- Les matrices sont moyennees sur les tetes, ce qui peut masquer des specialisations par tete.
- Les resultats ne couvrent que deux sequences locales.
- Les triangle attentions peuvent etre sensibles au choix d'extraction et de reduction de dimension.
- Les jobs et donnees brutes locales ne doivent pas etre publies tels quels si elles contiennent des sequences privees.

## Travaux A Faire

Pour transformer cette analyse en etude plus robuste:

1. Ajouter au moins 20-50 proteines de longueurs et topologies variees.
2. Comparer les liens d'attention aux contacts 3D derives des distances CA-CA.
3. Mesurer precision, rappel et enrichissement des liens d'attention pour predire les contacts.
4. Analyser les tetes separement au lieu de seulement utiliser la moyenne.
5. Ajouter des figures exportables: heatmaps, distributions d'entropie, graphes residue-residue.
6. Publier un script reproductible `scripts/analyze_attention.py`.

## Reproductibilite

Lancement de l'interface:

Voir la section *Installation* du [README](../README.md).

Extraction d'attention depuis l'interface:

1. Lancer une prediction.
2. Ouvrir le job termine.
3. Selectionner les familles d'attention.
4. Lancer l'analyse.
5. Explorer les heatmaps et la structure 3D.

Les sorties attendues par run:

```text
meta.json
pairformer_self_attn.npz
triangle_start.npz
triangle_end.npz
diffusion_token.npz
*_cube.npz
```

## Positionnement

OPENFOLD_PROJECT n'est pas un outil de validation biologique autonome. C'est un environnement d'interpretabilite pour observer comment OpenFold3-MLX distribue l'information entre residus pendant la prediction. Les resultats doivent etre croises avec les scores de confiance, les distances structurelles, les annotations biologiques et, si possible, des structures experimentales.
