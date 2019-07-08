Quora question pair study - distance measurements and word2vec
https://github.com/philopaszoon/ds/blob/master/quora_question_pairs/quora_pos_models.ipynb


What is word2vec

Word2vec evaluates the tendency of words to co-occur within a particular window (say, within 5 words).      Instead of producing a prediction, the algorithm instead returns the weights that would be used to arrive at that prediction.  Each word, then, is represented by a list of weights, each indicating the words participation in some aspect of meaning or sentence structure that would contribute to its odds of appearing in that place.  For example, word2vec vectors can capture the difference between verbs and nouns, the difference between male and female and the difference between positive and negative.  It is, for this reason, quite remarkable, especially considering the relatively simple way the result is achieved.  

Vectors can be trained to any desired length and it's likely that different lengths are optimal for different problems.  Most people seem to prefer vectors of between 300 and 400 weights.  Performance and memory are certainly concerns with these, but it's likely that most problems simply don't benefit from using vectors much longer than this.  

Preparing word2vec models

I used three different kinds of word2vec models in this project.  
1) a pre-trained model from Google
2) a pre-trained model from Spacy
3) a group of models trained on this quora data

Of these, the pre-trained models have the advantage of having been trained on very large volumes of data.  

However, they do not have many of the named entities that are present in this data.  I also wanted to explore word2vec models that included parts-of-speech and other information related to sentence structure.  

<hr>

Distance Measurements 

I explored two different kinds of distance measurement, Time Warped Distance and Cosine Similarity.  

Time Warped Distance measurement:
(https://en.wikipedia.org/wiki/Dynamic_time_warping)

This measurement takes the idea of euclidian distance a little further by seeking the shortest distance between each word in one sentence and any word in the other sentence being considered.  In a given pair of sentences [a b c] and [x y z], it can happen that point "a" in the first vector is closer to all the points "x", "y" and "z" in the other vector than either "b" or "c" is.  This measurements permits that, taking the shortest available distance even if some points get left out or get used multiple times.  This quality makes it good at getting the minimum possible distances between sentences.  It's also good at finding similarity between sentences that are similar but misaligned.  

Observe that euclidean distance does provide the base measurement between words themselves.  That is, the timeseries being considered here is not [x y z] but words in a sentence, so it's the explicitly the distance between words, not letters or any other abtraction, that make up the unit elements of the calculation.

There is another version of the function in the file that can use either correlation or cosine similarity as a base measurement instead.  Both versions are used, as I found that the version that uses euclidean distance is better at finding differences between sentences that are closely matched, while the one that uses correlation is better at finding differences between sentences that are a little further apart to begin with (see the named entity movement notebook).  


    def viDTWDistance(m1, m2):
        DTW={}
        for i in range(len(m1)): DTW[(i, -1)] = float('inf')
            for i in range(len(m2)): DTW[(-1, i)] = float('inf')
                DTW[(-1, -1)] = 0

                for i in range(len(m1)):
                    for j in range(len(m2)):
                        dist= (distance.euclidean(m1[i],m2[j]))**2
                        DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

                        longer = len(m1)
                        if(len(m2) > longer): longer = len(m2)
                        rval =  (sqrt(DTW[len(m1)-1, len(m2)-1]))/longer
        return rval


<hr>


Cosine Similarity:

This is the most commonly used method of measuring distance between vectors.  The basic principle underlying this measurement is that a vector, in any number of dimensions, can always be thought of as having a particular "direction".  While this is easy to visualize in two and three dimensions, the analogy remains valid in any number of dimensions.  But if each vector has an assignable direction, it much also be true that there is a measurable angle between them.  

So, how do we measure the "angle" between sentences?  The first thing to note is that sentences don't have their own vectors.  Word2vec gives us word vectors to play with, but there are relatively few domains in which whole sentences are used repatitively such that it would make sense to calculate vectors for them (such domains do exist, btw, but quora sentence pairs is certainly not one of them).  

Instead, we have to manufactor sentence vectors by combining word vectors.  Generally, this is done by simply averaging the word vectors from the words in each sentences down to a single vector for each sentence.  As noted in the introduction, this process blurs sentences and loses information. 

It has the advantage, however, of being fast.


    def angle_between(a,b):
        arccosInput = dot(a, b) / (norm(a) * norm(b))
        arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
        arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
        return arccosInput, math.acos(arccosInput)

<hr>


Overall, the idea of measuring a distance between sentences is a poor substitute for semantic comprehension.  Any process that tries to reduce the differences in meaning between sentences to a single distance measurement ignores the fact that sentences differ in meaning for a variety of reasons and in a variety of ways.   

For example, it's not hard to come up with two sentences that have the same meaning despite not sharing a single word (see the  introduction for an example).  Even an algorithm like Word Mover Distance, which I studied but did not use in this project, can only overcome this problem in a very limited way because it is, still, limited to making word-by-word comparisons.  

Cosine Similarity, notably, does not have that particular problem because all the words in the sentence are averaged together before the measurement is taken.  Cosine Similarity will also do better at picking up on similarities among aspects of words.  For example, a sentence discussing soldiers and a sentence discussing football players have in common that they both discuss masculine characters.  If the vectors are long enough and trained on enough data, Cosine Similarity should be able to pick up on that, whereas the other distance measurements would not.  

Nevertheless, all of these distance measurements are crude compared to the subtlety and complexity of the sentences they're trying to describe.  

Finally, as a general observation, distance measurements are more likely to see a match where there isn't one than to see a difference where there isn't one.  The image below, for example, shows matching sentences on the top line (1 in the Y axis) and sentences that don't match on the bottom line (0 in the Y axis).  Clearly, there are very few matching sentences below .5 in cosine similarity, while non-matching images are much better distributed.

Thus, the best way to improve the result on this task is to find ways to tease apart sentences that have small distance measurements (or, below, a small angle and high cosine similarity) but that, nevertheless, don't have the same meaning.  In other words, find specific elements in the sentences that indicate a non-match.   
<br> 
![cosine similarity](https://github.com/philopaszoon/ds/blob/master/quora_question_pairs/cos_scatter.png)

This is the approach I took in the named_entity_movement notebook.  In fact, this turned out to also be the most practical use of the Time Warped Distance measurement, but we'll get to that.

<hr>

Spacy distances:
<br>https://github.com/philopaszoon/ds/blob/master/quora_question_pairs/quora_spacy_distances.ipynb

I took advantage of built in Spacy vectors to explore a few more variations on the distance measurement idea, in particular removing stop words and lemmatizing words.    It's hard to say that these steps helped.  One problem may be that lemmatization outputs are not as consistent as might be hoped.   For example, Spacy often lemmatizes the word "good" into the word "well" which, to me, seems more ambiguous than the original word.  

What's more likely, however, is that lemmatization simply doesn't solve the underlying problem with distance measurements.  In the case of cosine similarity, it doesn't solve the blurring problem and in the case of time warped distance measurement, it doesn't solve the problem of dependence on word-by-word distance measurement (that is, stop words add a little bit of noise to the dynamic time warped measurement, but they don't stop the technique from discovering the best alignment of words between sentences).  


<hr> 

Spacy Reference Measurement:
<br>https://github.com/philopaszoon/ds/blob/master/quora_question_pairs/quora_spacy_reference_sims.ipynb

Spacy has its own internal sentence comparison feature.  In this notebook, I run and plot the output of that feature simply as a reference point.  The data is not included in the final learning algorithm.  As far as I can tell, this measurement is simply cosine similarity based on Spacy's own vectors.







