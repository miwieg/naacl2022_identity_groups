This repository contains all new resources we created for our NAACL 2022 paper "Identifying Implicitly Abusive Remarks about Identity Groups using a Linguistically Informed Approach" by Michael Wiegand, Elisabeth Eder and Josef Ruppenhofer.

The supplementary data for this research includes 4 directories and 2 files whose contents are briefly described below:

# file *supplementary_notes.pdf*
An additional document that specifies details regarding experiments carried out in our paper for which there was insufficient space in the paper.

# file *data_sheet.pdf*
A data sheet providing summarizing important information about the data contained in this repository.
 
# directory *LabelledSentences*
This directory contains the annotation of the English sentences (sentences.english.csv) and German sentences (sentences.german.csv) extracted from Twitter. These two files represent the central data of this research. Each file includes the annotation of both the main task (column "LABEL") and all component tasks (the name of the respective columns should be self-explanatory).

# directory *LexiconsForPerpetratorsOrNonConformistViews*
This directory contains the lexicon files we created for the detection of perpetrators (i.e. perpetrator-evoking verbs) and non-conformist views (i.e. fine-grained sentiment of the agent towards the patient). 
The directory includes both the lexicons manually created via crowdsourcing and the extensions that have been built by training a supervised classifier on these manually-compiled lexicons.
For the perpetrator-evoking verbs, we also included a file with the invented sentences that the crowdworkers produced.

# directory *Guidelines*
This directory contains the annotation guidelines for building the datasets as presented to the crowdworkers. Notice that the terminology used in the guidelines and the one in the paper may vary slightly since the crowdworkers were not trained linguistics. Therefore, they were not familiar with several technical terms, so we replaced some of them by more common terms (provided that they are nearly synonymous). For example, instead of "agent" and "patient", we refer to "subject" and "object", which, in the context of our annotation, amount the same thing.

# directory *Code*
A re-implementation of the linguistically informed classifier. The original code used some software which is not publicly available. In the re-implemented version, such software has been replaced by publicly available components. Overall, the performance of this version is comparable with the version used in the experiments of the paper.

# attribution
This data set is published under [Creative Commons Attribution 4.0](https://github.com/miwieg/naacl2022_identity_groups/edit/master//LICENSE).

# contact information
Please direct any questions that you have about this software to Michael Wiegand at University of Klagenfurt.

Michael Wiegand email: Michael.Wiegand@aau.at


[![DOI](https://zenodo.org/badge/479702958.svg)](https://zenodo.org/badge/latestdoi/479702958)
