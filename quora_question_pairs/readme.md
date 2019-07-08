Quora question pair study - introduction


Code descriptions are in the following sections:

[Preprocessing](https://github.com/philopaszoon/ds/blob/master/qfin/preprocessing.md)
|
[Adding features](https://github.com/philopaszoon/ds/blob/master/qfin/somefeatures.md)
|
[Libs](https://github.com/philopaszoon/ds/blob/master/qfin/splib.md)
|
[Word2vec](https://github.com/philopaszoon/ds/blob/master/qfin/w2v.md)
|
[Named Entities](https://github.com/philopaszoon/ds/blob/master/qfin/name_entities.md)
|
[Learning](https://github.com/philopaszoon/ds/blob/master/qfin/learning.md)

### The problem

When are two questions the same question?  This is a problem faced by Quora, which has incoming questions from users that, to avoid duplication, have to be matched against questions that have already been posted.

A good place to start is that two questions are the same when they have the same words.  However, it can (infrequently) happen that two sentences that have the same words mean different things, for example if the order is different.  It can also (much more frequently) happen that two sentences that have completely different words mean the same thing. For example:

<li> Are autodidacts smarter than other people?
<li> How intelligent do you have to be to teach yourself hard subjects?

These two questions do not share words but are, generally, the same question.  There's something about self learning and a question about about how smart one has to be to do it. If you think about the images that flash in your mind when you read these questions, they're probably the same or very close in both cases.

This particular pair of questions highlights some of the difficulties of taking statistical approaches to the problem of sentence comparison. For example, it's generally true that yes/no questions do not match questions that are not yes/no questions, but not in this case.  It's generally true that questions that share a lot of words match, while questions that share few words are less likely to match, but not in this case.  

Beyond comparing words, one can use word2vec vectors to try to capture and compare aspects of words. For example, word2vec might help you identify that a "car" is similar to an "auto" and to a "vehicle".  That’s useful if the problem you’re working with is sufficiently narrowly defined. For example, let’s look at two sentences:

<li> I drove my car to the airport.
<li> I drove my auto to the airport.

Are they the same?

Word2vec gives numeric representations of these words that can be used to summarize the sentences and measure a distance between them that can, potentially, indicate how similar they are. But that distance, without a context, doesn’t say very much unless it happens to be 0. Let’s look at a third sentence:

<li> I drove my mother to the airport.

Now we have a context. The word2vec vector for “mother” is going to be quite distant from the vectors for “car” and “auto”.  Clearly, 1 and 2 are more similar to each other than 1 and 3.

Let’s say the measured distance between 1 and 2 is .5. Are we done? Are all sentences that have a measured distance lower than .5 the same? As it turns out, no. Here are two more sentences:

<li> What’s the weather like in California in the winter?
<li> What’s the weather like in Florida in the winter?

The answers to these two questions might be about the same, but the questions are obviously not.  Nevertheless, the measured distances (we’ll look at a few different approaches) will in most cases find these two sentences to be very similar.

After all, are California and Florida really that much different from each other than “car” and “auto”? California and Florida are both states. “Car” and “auto” are both vehicles. In both cases, they are children of a common parent class that can, more or less, appear interchangeably in sentences. Since that’s all word2vec looks at (the ability to appear interchangeably in a sentence), it’s going to miss the distinction that we find obvious, that “car” and “auto” really are interchangeable representations of a common object while California and Florida are proper nouns naming distinct objects.

Another problem with distance measurements is that they require some sort of aggregation of the sentence. For example, to measure the cosine similarity between two sentences, one would begin by averaging together the word vectors in each sentence so that there are only two remaining vectors (one for each sentence) left to compare. 

But what actually happens when you average together the word vectors for the sentence, “the dog slept under the tree.”? The resulting single vector is a blur that would not do a better job of conveying the meaning of the sentence than pictures of a dog, a tree and someone sleeping would if they were blurred together. Further, the longer and more complex the sentence, the more information is lost in the blurring.

Thus, distance measurements provide limited information.  In spite of that, this project is largely devoted to exploring them.

---------------------------------------------------


Further Challenges:

There are a number of difficulties with this  task, some inherent and some related to the quality of the data.

Inherently, sentences are mere collections of symbols that require an external model to be understood.  If I tell you something about a chair, you only know what I'm talking about because you know what a chair is from experience.  You've seen chairs, you've sat on them, you know them as commodity items that can be purchased and as art, and so on.  You have a model, in your mind, of what a chair is.  Functionally, it's essentially a Platonic model, although we don't have to buy into the strong version of that idea to recognize its relevance.

Similarly, if I tell you something that involves the concept of fairness, you can refer to an existing model of the meaning of that word which comes from your experience, such as the emotions stirred by betrayal.   The success of language as a medium of exchange of ideas is a testament to the commonality of experience, but it's also clear that the experience basis is a pre-requisite for that success.   

It may be possible to bootstrap an experience-model from language data alone.  One might do so, for example, by looking for phrases that directly report experience ("I feel betrayed", "I sat on a chair" ..), but such an effort is far beyond the scope of these exercises.  <i>[suggestion for further study - explore vectors of words that appear in the context of direct experience reports]</i>

Word2vec vectors are the best available substitute for an experience basis.  The vector for "chair" can tell you that it's a tangible object (more correctly, that it's more like other tangible objects than it is like any particular intangible objects), but only if you explicitly query that aspect of it.  An experience basis is an effective bootstrap for learning because it _favors_ some queries, namely those that can be resolved by referring to a direct experience.  That is, "can I touch it?" is probably among the first things you would ask about a new word that you don't know the meaning of.  

But the vector, by itself, doesn't tell you what questions to ask.   As far as the vector is concerned, the fact that a word is gendered (or not) is every bit as important as the fact that the thing it refers to is tangible (or not).  One might consider limiting the number of neurons in a word2vec training process to try to force a prioritization of questions (20?), but it's not obvious that inferences made from co-occurrence would lead to the same priorities that we arrive at from our experience.    That is, among the 20 neurons in such a limited training, there probably isn't going to be one for "Is it bigger than a bread-box?".  <i>[suggestion for further study - train a word2vec model with large vectors (over 1000) and explore the prospect of selecting particular vectors for particular discrimination tasks.  Can the best 20 questions be identified?]</i>

-----------------------------

Scoring:

The scoring is often questionable.  Take the following examples:

<li> "What type of government does Guatemala have? How does it compare to the one in Canada?"
<li> "What type of government does Guatemala have? How does it compare to the one in Venezuela?"

It may seem obvious that these are different questions because Canada is not Venezuela, but the scoring on questions like these is very inconsistent.  It's not even obvious how to quantify the inconsistency, since it isn't always as simple as a named entity difference.  One simply has to spend time with the data to note how often one finds oneself arguing with the scoring.  Personally, I feel that somewhere between 10% and 15% of the question pairs are wrongly scored, but that also includes a large number that are genuinely hard to score.  Because of this, I would be suspicious of any algorithm that got much better than 80% accuracy, no matter how sophisticated it may be.

-----------------------------

Finally, before we start looking at code, I want to ask the most general question that we can ask about this data.

--  What is a sentence?

Narrowly, it's just a collection of symbols connected to each other by rules.  But the idea of a sentence can also be defined in terms of its purpose.   A sentence is always a narrative.  There are things (generally represented by nouns) and some kind of action relating to them (generally represented by verbs).  Every sentence tells a story of some kind about the things in it and the events that relate to those things.

Often, especially with questions, the event (the verb) is simply "to be".  

<li> What [is] your favorite color?

.. which could be rewritten declaratively as "Your favorite color [is] what".    The essential narrative of the sentence, then, is simply "color [is] what".    The fact that the object of the verb is a placeholder ("what") is what really marks this as a question.

We can think of every verb as a kind of black box that provides the transformation that gives rise to the narrative.  In this case, the transformation is really an equation.
<li> x            [=]    b
<li> color   [is]    what

But real world sentences are not this simple.  Consider a sentence I just used:

<li> We can think of every verb as a kind of black box that provides 
     the transformation that gives rise to the narrative.

The verbs are "think", "provides" and "gives".  But how would you model the action implied by the word, "provides", mathematically?  It's not an easy question.  Further, the word "provides" typically prefers to have an indirect object.  Something is provided "to" something or someone.  Here, the indirect object would have to be the person interpreting the words, but it's not explicit.  

In the case of the phrase "black box provides the transformation", the direct object, "transformation", is the thing that is transformed.  It's "provided" to someone, whatever that means.  Made available?  Given?  Relocated?  How can one model this transformation?  

Once again it's tempting to ask if word2vec can help with this problem, and that's a question I hope to take up in the future. <i>[suggestion for further study - can vectors be used to classify verbs in terms of the transformation they catalyze?  Can one predict, for example, that a "dropped" object is likely to eventually "land"?  Can we say, as above, that if something is "provided" to someone, that that person is thereafter somehow enriched by it?  How are they enriched? ]</i>











 
  

