Quora question pair study - Named Entities

quora_named_entity_counts_cm:
<br> https://github.com/philopaszoon/ds/blob/master/quora_question_pairs/quora_named_entity_counts_cm.ipynb

The nature of language is such that small details can have a significant impact on the meaning of a sentence.  In this notebook, we take on one class of those details, named entities.  The objective is to pay special attention to the difference between sentences specifically where those differences are due to unmatched named entities. 

This turns out to be a good use of the Dynamic Time Warp Distance measure, in part because the rotation (a term that will make sense after you look at the notebook) is almost completly obscured by the averaging which takes place as part of the cosine similarity measurement and in part because the TWDistance measurement is not overly sensative to word placement, so that as long as the same word gets the same rotation in both sentences, they'll be a match.

This step is fairly long and there's a lot going on, so instead of repeating it all here I'll ask you to read the notebook itself.  One thing I do want to point out is that by examining the results of this process, it's possible get some tips on what further steps can be taken to identify non-matching sentences.  The bottom of the notebook covers that a bit.

