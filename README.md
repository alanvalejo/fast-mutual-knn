## MkNN

**About**

This is an alternative Python implementation of graph construction method MkNN (Mutual k Nearest Neighbor Graph Construction), used in Berton and Lopes (2016) [1]. The implementation is based on Kd-tree and Multi-threading and is faster than the original described in the paper, specially for large data sets.

**Usage**

> MkNN execution

    python -OO mknn.pyo -f input/square.dat -k 3

> Input: any numerical dataset with any delimiter for attributes

> Output: a weighted undirected graph in the format: filename + '-gbili.ncol'


**Parameters**

| Option					| Domain					| Required	| Default	| Description															|
|:------------------------- |:------------------------- | --------- | --------- |:--------------------------------------------------------------------- |
| -f, --filename			| string [FILE]				| yes		| -			| Dataset as input file													|
| -o, --output				| string [FILE]				| no		| ncol		| Output file															|
| -k, --k					| [1,n] Integer interval	| no		| 3			| k for mutual kNN														|
| -t, --threads				| [0,n] Integer interval	| no		| 4			| Number of  threads													|
| -e, --format				| ['ncol', 'pajek']			| no		| ncol		| Format output file													|
| -c, --skip_last_column	| bool						| no		| true		| Skip the last column													|

**Dependencies**

* Python: tested with version 2.7.13.
* Packages needed: numpy, scipy and multiprocessing.

**Known Bugs**

Please contact the author for problems and bug report.

**Contact**

* Alan Valejo.
* Ph.D. candidate at University of São Paolo (USP), Brazil.
* alanvalejo@gmail.com.

**License**

* Can be used for creating unlimited applications
* Can be distributed in binary or object form only
* Non-commercial use only
* Can modify source-code and distribute modifications (derivative works)
* Attribution to software creator must be made as specified: Giving credit to the author by citing the paper [1] or source [2]

**References**

> [1] Berton, Lilian; Faleiros, T.; Valejo, A.; Valverde-Rebaza, J.; and Lopes, A. A.: RGCLI: Robust Graph that Considers Labeled Instances for Semi-Supervised Learning. Neurocomputing, (2016).

> [2] https://github.com/alanvalejo/rgcli

~~~~~{.bib}
@article{Berton_2016,
    author={Lilian Berton and Thiago de Paulo Faleiros and Alan Valejo and
    Jorge Valverde-Rebaza and Alneu de Andrade Lopes},
    title={RGCLI: Robust Graph that Considers Labeled Instances for Semi-Supervised Learning},
    journal={Neurocomputing},
    year={2016}
}
~~~~~

<div class="footer"> &copy; Copyright (C) 2016 Alan Valejo &lt;alanvalejo@gmail.com&gt; All rights reserved.</div>
