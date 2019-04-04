Quora question pair study - preprocessing


quora_cleanup_1:
<br>https://github.com/philopaszoon/capstone1/blob/master/qfin/quora_cleanup_1.ipynb
* using tabs instead of commas to separate columns
* cleaning up the IDs
* assigning group IDs to exact duplicates and metagroup IDs to questions that are scored as matches
* assigning keys to lines (linekey) to make it easy to merge data downstream 
* dropping questions that have very few words after stop words are removed
* removing questions with non-english characters
* saving the data to a mysql database


quora_contractions_3:
<br>https://github.com/philopaszoon/capstone1/blob/master/qfin/quora_contractions_3.ipynb
* expanding contractions [ he's -> he is,   won't -> will not,   etc ]
* note that possessive contractions and contractions in named entities (which are mostly possessive as well) are not expanded


